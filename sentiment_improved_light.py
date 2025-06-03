# -*- coding: utf-8 -*-
"""
Versión mejorada y optimizada del script Sentiment.ipynb
Mantiene el enfoque original pero implementa mejoras para aumentar la precisión
con menor consumo de recursos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string
import re

# Descargar recursos NLTK necesarios
nltk.download('stopwords', quiet=True)

# Cargar los datos - usar el dataset original para mantener la misma base
print("Cargando datos...")
data = pd.read_csv('Sentiment(1) 1.csv')
data = data[['text', 'sentiment']]

# Filtrar datos nulos
data = data.dropna(subset=['text', 'sentiment'])

# Reducir tamaño del dataset para procesamiento más rápido (MEJORA 1)
# Usar una muestra estratificada para mantener la proporción de clases
data = data.groupby('sentiment').apply(lambda x: x.sample(min(len(x), 1000), random_state=42)).reset_index(drop=True)

# Dividir en entrenamiento y prueba
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])

# Filtrar sentimientos neutrales como en el original
train = train.query("sentiment != 'Neutral'")
test = test.query("sentiment != 'Neutral'")

print(f"Datos de entrenamiento: {train.shape[0]} tweets")
print(f"Datos de prueba: {test.shape[0]} tweets")

# Función para preprocesar texto (mejorada pero simplificada)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs, menciones, hashtags y RT como en el original
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+|rt', '', text)
    
    # Eliminar signos de puntuación (mejora)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Eliminar números (mejora)
    text = re.sub(r'\d+', '', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Separar en palabras y filtrar stopwords
    words = [w for w in text.split() if len(w) >= 3 and w not in stopwords.words('english')]
    
    return ' '.join(words)

# Preprocesar los textos
print("Preprocesando textos...")
train['processed_text'] = train['text'].apply(preprocess_text)
test['processed_text'] = test['text'].apply(preprocess_text)

# Separar por sentimiento como en el original
train_pos = train[train['sentiment'] == 'Positive']
train_pos_text = train_pos['processed_text']
train_neg = train[train['sentiment'] == 'Negative']
train_neg_text = train_neg['processed_text']

# Visualizar nubes de palabras (tamaño reducido para menor uso de memoria)
def wordcloud_draw(words, color='black', filename=None):
    words = ' '.join(words)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=800,  # Reducido de 2500
                      height=600  # Reducido de 2000
                     ).generate(words)
    plt.figure(figsize=(8, 6))  # Reducido de (13, 13)
    plt.imshow(wordcloud)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename)
        print(f'Wordcloud saved as {filename}')
    plt.close()

print("Generando nubes de palabras...")
print("Palabras positivas")
wordcloud_draw(train_pos_text, 'white', 'positive_wordcloud_light.png')
print("Palabras negativas")
wordcloud_draw(train_neg_text, 'black', 'negative_wordcloud_light.png')

# Preparar datos para NLTK (manteniendo el enfoque original)
tweets = []
for index, row in train.iterrows():
    words = row.processed_text.split()
    tweets.append((words, row.sentiment))

# Extraer características (reduciendo el número para menor uso de memoria)
def get_words_in_tweets(tweets):
    all_words = []
    for (words, _) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    # Reducir a 500 palabras más frecuentes (en lugar de 2000)
    word_features = list(wordlist.keys())[:500]
    return word_features

print("Extrayendo características...")
w_features = get_word_features(get_words_in_tweets(tweets))

# Función de extracción de características simplificada
def extract_features(document):
    document_words = set(document)
    features = {}
    
    # Características unigrama (como en el original)
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    
    # Solo algunas características adicionales simples
    features['length'] = len(document)
    
    return features

# Visualizar nube de palabras de características
wordcloud_draw(w_features, 'gray', 'features_wordcloud_light.png')

print("Preparando datos para clasificación...")
# Preparar conjunto de entrenamiento
training_set = nltk.classify.apply_features(extract_features, tweets)

# Entrenar clasificador Naive Bayes original
print("\nEntrenando clasificador Naive Bayes (Original)...")
classifier_nb = nltk.NaiveBayesClassifier.train(training_set)
print("Características más informativas:")
print(classifier_nb.most_informative_features(10))

# Entrenar clasificador MultinomialNB (solo uno adicional para reducir carga)
print("\nEntrenando clasificador MultinomialNB...")
classifier_mnb = SklearnClassifier(MultinomialNB())
classifier_mnb.train(training_set)

# Preparar datos de prueba
test_pos = test[test['sentiment'] == 'Positive']
test_pos_text = test_pos['processed_text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg_text = test_neg['processed_text']

# Función para evaluar un clasificador
def evaluate_classifier(classifier, name):
    print(f"\nEvaluando clasificador {name}...")
    
    # Evaluar en datos negativos
    neg_correct = 0
    for obj in test_neg_text:
        words = obj.split()
        if classifier.classify(extract_features(words)) == 'Negative':
            neg_correct += 1
    
    # Evaluar en datos positivos
    pos_correct = 0
    for obj in test_pos_text:
        words = obj.split()
        if classifier.classify(extract_features(words)) == 'Positive':
            pos_correct += 1
    
    # Calcular precisión
    neg_accuracy = neg_correct / len(test_neg_text) if len(test_neg_text) > 0 else 0
    pos_accuracy = pos_correct / len(test_pos_text) if len(test_pos_text) > 0 else 0
    overall_accuracy = (neg_correct + pos_correct) / (len(test_neg_text) + len(test_pos_text))
    
    print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_accuracy:.2%})")
    print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_accuracy:.2%})")
    print(f"Precisión general: {overall_accuracy:.2%}")
    
    return overall_accuracy

# Evaluar clasificadores
print("\n--- Evaluación de Clasificadores ---")
nb_accuracy = evaluate_classifier(classifier_nb, "Naive Bayes (Original)")
mnb_accuracy = evaluate_classifier(classifier_mnb, "MultinomialNB")

# Implementar votación simple entre los dos clasificadores
print("\n--- Clasificador Ensemble Simplificado ---")
def ensemble_classify(features):
    votes = []
    votes.append(classifier_nb.classify(features))
    votes.append(classifier_mnb.classify(features))
    
    # Retornar el voto mayoritario
    if votes.count('Positive') > votes.count('Negative'):
        return 'Positive'
    else:
        return 'Negative'

# Evaluar el clasificador ensemble
neg_correct = 0
for obj in test_neg_text:
    words = obj.split()
    if ensemble_classify(extract_features(words)) == 'Negative':
        neg_correct += 1

pos_correct = 0
for obj in test_pos_text:
    words = obj.split()
    if ensemble_classify(extract_features(words)) == 'Positive':
        pos_correct += 1

ensemble_accuracy = (neg_correct + pos_correct) / (len(test_neg_text) + len(test_pos_text))

print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_correct/len(test_neg_text):.2%})")
print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_correct/len(test_pos_text):.2%})")
print(f"Precisión general: {ensemble_accuracy:.2%}")

# Comparar los resultados con un gráfico simple
print("\n--- Comparación de Clasificadores ---")
classifiers = ["Naive Bayes", "MultinomialNB", "Ensemble"]
accuracies = [nb_accuracy, mnb_accuracy, ensemble_accuracy]

# Visualizar la comparación
plt.figure(figsize=(8, 5))
plt.bar(classifiers, accuracies, color=['blue', 'green', 'orange'])
plt.title('Comparación de Precisión entre Clasificadores')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('classifier_comparison_light.png')
print("Gráfico de comparación guardado como 'classifier_comparison_light.png'")

# Resumen de mejoras implementadas
print("\n--- Resumen de Mejoras Implementadas ---")
print("1. Preprocesamiento mejorado del texto")
print("2. Selección de características más relevantes")
print("3. Características adicionales simples (longitud)")
print("4. Uso de clasificador MultinomialNB de scikit-learn")
print("5. Votación entre clasificadores (ensemble)")

print("\nProceso completado.")
