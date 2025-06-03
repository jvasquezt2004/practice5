# -*- coding: utf-8 -*-
"""
Versión del script original para comparación directa con el mejorado
Usa el dataset original y solo el clasificador Naive Bayes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Descargar stopwords si es necesario
nltk.download('stopwords', quiet=True)

print("=== ANÁLISIS DE SENTIMIENTO ORIGINAL ===")
print("Cargando datos originales...")
# Usar el dataset original
data = pd.read_csv('Sentiment(1) 1.csv')
data = data[['text', 'sentiment']]

# Filtrar datos nulos
data = data.dropna(subset=['text', 'sentiment'])

print("\nDistribución de sentimientos en el dataset original:")
sentiment_distribution = data['sentiment'].value_counts()
print(sentiment_distribution)

# Dividir en entrenamiento y prueba
print("\nDividiendo en conjuntos de entrenamiento y prueba...")
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])

# Filtrar sentimientos neutrales usando pandas query method
train = train.query("sentiment != 'Neutral'")
test = test.query("sentiment != 'Neutral'")

print(f"Datos de entrenamiento: {train.shape[0]} tweets")
print(f"Datos de prueba: {test.shape[0]} tweets")

train_pos = train[train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[train['sentiment'] == 'Negative']
train_neg = train_neg['text']

# Preparar datos para NLTK
print("\nPreparando datos para clasificación...")
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

test_pos = test[test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg = test_neg['text']

def get_words_in_tweets(tweets):
    all = []
    for (words, _) in tweets:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

print("Extrayendo características...")
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features

print("Entrenando clasificador Naive Bayes (Original)...")
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Características más informativas:")
print(classifier.most_informative_features(15))

# Evaluar en datos negativos
print("\nEvaluando clasificador...")
neg_correct = 0
neg_total = len(test_neg)
for obj in test_neg:
    # Aplicar el mismo preprocesamiento que en el entrenamiento
    words_filtered = [e.lower() for e in obj.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    
    res = classifier.classify(extract_features(words_without_stopwords))
    if(res == 'Negative'):
        neg_correct = neg_correct + 1

# Evaluar en datos positivos
pos_correct = 0
pos_total = len(test_pos)
for obj in test_pos:
    # Aplicar el mismo preprocesamiento que en el entrenamiento
    words_filtered = [e.lower() for e in obj.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    
    res = classifier.classify(extract_features(words_without_stopwords))
    if(res == 'Positive'):
        pos_correct = pos_correct + 1

neg_accuracy = neg_correct / neg_total if neg_total > 0 else 0
pos_accuracy = pos_correct / pos_total if pos_total > 0 else 0
overall_accuracy = (neg_correct + pos_correct) / (neg_total + pos_total)

print("\n--- Resultados del clasificador original ---")
print(f"[Negative]: {neg_correct}/{neg_total} ({neg_accuracy:.2%})")
print(f"[Positive]: {pos_correct}/{pos_total} ({pos_accuracy:.2%})")
print(f"Precisión general: {overall_accuracy:.2%}")

# Visualizar resultados
plt.figure(figsize=(10, 6))
metrics = ['Precisión general', 'Precisión negativos', 'Precisión positivos']
values = [overall_accuracy, neg_accuracy, pos_accuracy]
colors = ['blue', 'red', 'green']

plt.bar(metrics, values, color=colors)
plt.title('Métricas del Clasificador Naive Bayes Original')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('naive_bayes_metrics_original.png')
print("\nGráfico de métricas guardado como 'naive_bayes_metrics_original.png'")

# Ejemplos de clasificación
print("\n=== EJEMPLOS DE CLASIFICACIÓN ===")
def classify_example(text):
    # Preprocesar el texto de ejemplo como en el entrenamiento
    words_filtered = [e.lower() for e in text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    
    # Clasificar
    features = extract_features(words_without_stopwords)
    sentiment = classifier.classify(features)
    prob_dist = classifier.prob_classify(features)
    
    print(f"Texto: '{text}'")
    print(f"Sentimiento: {sentiment}")
    print(f"Confianza: {prob_dist.prob(sentiment):.2%}")
    print("---")
    
    return sentiment

# Usar los mismos ejemplos que en el script mejorado
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
