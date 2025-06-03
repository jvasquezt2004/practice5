# -*- coding: utf-8 -*-
"""
Versión mejorada y simplificada del script Sentiment.py
Usa el dataset con oversampling puro (sin reducir clases) y solo el clasificador Naive Bayes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import string
import re

from wordcloud import WordCloud, STOPWORDS

# Descargar recursos NLTK necesarios
nltk.download('stopwords', quiet=True)

print("=== ANÁLISIS DE SENTIMIENTO MEJORADO (VERSIÓN SIMPLE) ===")
print("Cargando datos desde el dataset preprocesado con oversampling...")
# Usar el dataset preprocesado con oversampling
data = pd.read_csv('sentiment_processed_oversampled.csv')

# Verificar la distribución de clases
print("\nDistribución de sentimientos en el dataset:")
sentiment_distribution = data['sentiment'].value_counts()
print(sentiment_distribution)

# Dividir en entrenamiento y prueba (manteniendo estratificación)
print("\nDividiendo en conjuntos de entrenamiento y prueba...")
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])

print(f"Datos de entrenamiento: {train.shape[0]} tweets")
print(f"Datos de prueba: {test.shape[0]} tweets")

# Separar por sentimiento como en el original
train_pos = train[train['sentiment'] == 'Positive']
train_pos_text = train_pos['processed_text']
train_neg = train[train['sentiment'] == 'Negative']
train_neg_text = train_neg['processed_text']

# Generar nubes de palabras para visualización
def wordcloud_draw(words, color='black', filename=None):
    # Asegurar que todos los elementos sean strings
    words = [str(word) for word in words if not pd.isna(word)]
    words = ' '.join(words)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=800,
                      height=600
                     ).generate(words)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename)
        print(f'Wordcloud guardada como {filename}')
    plt.close()

print("\nGenerando nubes de palabras...")
print("Palabras positivas")
wordcloud_draw(train_pos_text, 'white', 'positive_wordcloud_simple.png')
print("Palabras negativas")
wordcloud_draw(train_neg_text, 'black', 'negative_wordcloud_simple.png')

# Preparar datos para NLTK (usando texto ya preprocesado)
print("\nPreparando datos para clasificación...")
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words = str(row.processed_text).split()
    # Filtrar stopwords y palabras muy cortas si aún quedan
    words_filtered = [word for word in words if len(word) >= 3 and word not in stopwords_set]
    tweets.append((words_filtered, row.sentiment))

# Extraer características
def get_words_in_tweets(tweets):
    all_words = []
    for (words, _) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    # Usar las 2000 palabras más frecuentes
    word_features = list(wordlist.keys())[:2000]
    return word_features

print("Extrayendo características...")
w_features = get_word_features(get_words_in_tweets(tweets))

# Función de extracción de características mejorada
def extract_features(document):
    document_words = set(document)
    features = {}
    
    # Características unigrama (como en el original)
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    
    # Características adicionales simples
    features['length'] = len(document)
    
    # Características de presencia de palabras positivas/negativas comunes
    # (basadas en las palabras más informativas encontradas anteriormente)
    positive_words = ['balanced', 'fair', 'job', 'good', 'great', 'best']
    negative_words = ['bad', 'worst', 'lose', 'corrupt', 'terrible']
    
    features['has_positive_word'] = any(word in document_words for word in positive_words)
    features['has_negative_word'] = any(word in document_words for word in negative_words)
    
    return features

# Visualizar nube de palabras de características
wordcloud_draw(w_features, 'gray', 'features_wordcloud_simple.png')

# Preparar conjunto de entrenamiento
training_set = nltk.classify.apply_features(extract_features, tweets)

# Entrenar clasificador Naive Bayes mejorado
print("\nEntrenando clasificador Naive Bayes mejorado...")
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Características más informativas:")
print(classifier.most_informative_features(15))

# Preparar datos de prueba
test_pos = test[test['sentiment'] == 'Positive']
test_pos_text = test_pos['processed_text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg_text = test_neg['processed_text']

# Función para evaluar el clasificador
def evaluate_classifier():
    print("\nEvaluando clasificador...")
    
    # Evaluar en datos negativos
    neg_correct = 0
    for obj in test_neg_text:
        words = str(obj).split()
        # Filtrar stopwords como en el entrenamiento
        words_filtered = [word for word in words if len(word) >= 3 and word not in stopwords_set]
        if classifier.classify(extract_features(words_filtered)) == 'Negative':
            neg_correct += 1
    
    # Evaluar en datos positivos
    pos_correct = 0
    for obj in test_pos_text:
        words = str(obj).split()
        # Filtrar stopwords como en el entrenamiento
        words_filtered = [word for word in words if len(word) >= 3 and word not in stopwords_set]
        if classifier.classify(extract_features(words_filtered)) == 'Positive':
            pos_correct += 1
    
    # Calcular precisión
    neg_accuracy = neg_correct / len(test_neg_text) if len(test_neg_text) > 0 else 0
    pos_accuracy = pos_correct / len(test_pos_text) if len(test_pos_text) > 0 else 0
    overall_accuracy = (neg_correct + pos_correct) / (len(test_neg_text) + len(test_pos_text))
    
    print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_accuracy:.2%})")
    print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_accuracy:.2%})")
    print(f"Precisión general: {overall_accuracy:.2%}")
    
    return overall_accuracy, neg_accuracy, pos_accuracy

# Evaluar clasificador
overall_accuracy, neg_accuracy, pos_accuracy = evaluate_classifier()

# Visualizar resultados
plt.figure(figsize=(10, 6))
metrics = ['Precisión general', 'Precisión negativos', 'Precisión positivos']
values = [overall_accuracy, neg_accuracy, pos_accuracy]
colors = ['blue', 'red', 'green']

plt.bar(metrics, values, color=colors)
plt.title('Métricas del Clasificador Naive Bayes Mejorado')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('naive_bayes_metrics_simple.png')
print("\nGráfico de métricas guardado como 'naive_bayes_metrics_simple.png'")

# Ejemplos de clasificación manual
print("\n=== EJEMPLOS DE CLASIFICACIÓN ===")
def classify_example(text):
    # Preprocesar el texto de ejemplo
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+|rt', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenizar y filtrar
    words = [word for word in text.split() if len(word) >= 3 and word not in stopwords_set]
    
    # Clasificar
    features = extract_features(words)
    sentiment = classifier.classify(features)
    prob_dist = classifier.prob_classify(features)
    
    print(f"Texto: '{text}'")
    print(f"Sentimiento: {sentiment}")
    print(f"Confianza: {prob_dist.prob(sentiment):.2%}")
    print("---")
    
    return sentiment

# Algunos ejemplos para probar
examples = [
    "I really loved the debate, very informative and well organized",
    "The candidates were terrible, they just attacked each other",
    "Not sure what to think about the debate, some good points raised",
    "This was the best political event I've seen in years",
    "Complete disaster, waste of time watching these people"
]

print("\nClasificando ejemplos...")
for example in examples:
    classify_example(example)

print("\nProceso completado exitosamente.")
