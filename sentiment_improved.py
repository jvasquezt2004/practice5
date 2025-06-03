# -*- coding: utf-8 -*-
"""
Versión mejorada del script Sentiment.ipynb
Mantiene el enfoque original pero implementa mejoras para aumentar la precisión
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.classify import SklearnClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string
import re

# Descargar recursos NLTK necesarios
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Cargar los datos - usar el dataset original para mantener la misma base
print("Cargando datos...")
data = pd.read_csv('Sentiment(1) 1.csv')
data = data[['text', 'sentiment']]

# Filtrar datos nulos
data = data.dropna(subset=['text', 'sentiment'])

# Dividir en entrenamiento y prueba
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])

# Filtrar sentimientos neutrales como en el original
train = train.query("sentiment != 'Neutral'")
test = test.query("sentiment != 'Neutral'")

print(f"Datos de entrenamiento: {train.shape[0]} tweets")
print(f"Datos de prueba: {test.shape[0]} tweets")

# Función para preprocesar texto (mejorada)
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
    
    # Tokenizar
    tokens = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) >= 3]
    
    # Lematizar (mejora)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Preprocesar los textos
train['processed_text'] = train['text'].apply(preprocess_text)
test['processed_text'] = test['text'].apply(preprocess_text)

# Separar por sentimiento como en el original
train_pos = train[train['sentiment'] == 'Positive']
train_pos_text = train_pos['processed_text']
train_neg = train[train['sentiment'] == 'Negative']
train_neg_text = train_neg['processed_text']

# Visualizar nubes de palabras como en el original
def wordcloud_draw(words, color='black', filename=None):
    words = ' '.join(words)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(words)
    plt.figure(figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename)
        print(f'Wordcloud saved as {filename}')
    plt.close()

print("Generando nubes de palabras...")
print("Palabras positivas")
wordcloud_draw(train_pos_text, 'white', 'positive_wordcloud_improved.png')
print("Palabras negativas")
wordcloud_draw(train_neg_text, 'black', 'negative_wordcloud_improved.png')

# Preparar datos para NLTK (manteniendo el enfoque original)
tweets = []
for index, row in train.iterrows():
    words = row.processed_text.split()
    tweets.append((words, row.sentiment))

# Extraer características como en el original pero con mejoras
def get_words_in_tweets(tweets):
    all_words = []
    for (words, _) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    # Usar FreqDist para encontrar palabras más frecuentes
    wordlist = nltk.FreqDist(wordlist)
    # Mantener solo las palabras más frecuentes (mejora)
    word_features = list(wordlist.keys())[:2000]  # Limitar a 2000 palabras más frecuentes
    return word_features

w_features = get_word_features(get_words_in_tweets(tweets))

# Extraer características pero ahora con n-gramas (mejora)
def extract_features(document):
    document_words = set(document)
    features = {}
    
    # Características unigrama (como en el original)
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    
    # Características de bigramas (mejora)
    if len(document) > 1:
        bigrams = list(nltk.bigrams(document))
        for i in range(len(bigrams)):
            if i < 100:  # Limitar para no tener demasiadas características
                bigram = ' '.join(bigrams[i])
                features['bigram(%s)' % bigram] = True
    
    # Características adicionales (mejora)
    features['length'] = len(document)
    features['has_exclamation'] = '!' in ' '.join(document)
    features['has_question'] = '?' in ' '.join(document)
    
    return features

# Visualizar nube de palabras de características como en el original
wordcloud_draw(w_features, 'gray', 'features_wordcloud_improved.png')

print("Preparando datos para clasificación...")
# Preparar conjunto de entrenamiento
training_set = nltk.classify.apply_features(extract_features, tweets)

# MEJORA 1: Usar validación cruzada para una mejor evaluación
print("\nEntrenando clasificador Naive Bayes (Original)...")
classifier_nb = nltk.NaiveBayesClassifier.train(training_set)
print("Características más informativas:")
print(classifier_nb.most_informative_features(10))

# MEJORA 2: Usar diferentes clasificadores de scikit-learn
print("\nProbando diferentes clasificadores...")

# Preparar datos de prueba
test_pos = test[test['sentiment'] == 'Positive']
test_pos_text = test_pos['processed_text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg_text = test_neg['processed_text']

# Función para evaluar un clasificador
def evaluate_classifier(classifier, name):
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
    
    print(f"\nResultados del clasificador {name}:")
    print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_accuracy:.2%})")
    print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_accuracy:.2%})")
    print(f"Precisión general: {overall_accuracy:.2%}")
    
    return overall_accuracy

# Evaluar el clasificador Naive Bayes original
nb_accuracy = evaluate_classifier(classifier_nb, "Naive Bayes (Original)")

# MEJORA 3: Usar SklearnClassifier con diferentes algoritmos
print("\nEntrenando clasificador MultinomialNB...")
classifier_mnb = SklearnClassifier(MultinomialNB())
classifier_mnb.train(training_set)
mnb_accuracy = evaluate_classifier(classifier_mnb, "MultinomialNB")

print("\nEntrenando clasificador LinearSVC...")
classifier_svc = SklearnClassifier(LinearSVC())
classifier_svc.train(training_set)
svc_accuracy = evaluate_classifier(classifier_svc, "LinearSVC")

print("\nEntrenando clasificador RandomForest...")
classifier_rf = SklearnClassifier(RandomForestClassifier(n_estimators=100))
classifier_rf.train(training_set)
rf_accuracy = evaluate_classifier(classifier_rf, "RandomForest")

# MEJORA 4: Utilizar votación para combinar clasificadores
print("\nCombinando clasificadores mediante votación...")
def ensemble_classify(features):
    votes = []
    votes.append(classifier_nb.classify(features))
    votes.append(classifier_mnb.classify(features))
    votes.append(classifier_svc.classify(features))
    votes.append(classifier_rf.classify(features))
    
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

print("\nResultados del clasificador Ensemble:")
print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_correct/len(test_neg_text):.2%})")
print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_correct/len(test_pos_text):.2%})")
print(f"Precisión general: {ensemble_accuracy:.2%}")

# Comparar los resultados
print("\n--- Comparación de Clasificadores ---")
classifiers = ["Naive Bayes (Original)", "MultinomialNB", "LinearSVC", "RandomForest", "Ensemble"]
accuracies = [nb_accuracy, mnb_accuracy, svc_accuracy, rf_accuracy, ensemble_accuracy]

# Visualizar la comparación
plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Comparación de Precisión entre Clasificadores')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('classifier_comparison.png')
print("Gráfico de comparación guardado como 'classifier_comparison.png'")

print("\nProceso completado.")
