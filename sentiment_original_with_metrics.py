# -*- coding: utf-8 -*-
"""
Versión ligeramente modificada del script original Sentiment.ipynb
Solo se han añadido métricas para poder comparar con la versión mejorada
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Descargar stopwords si es necesario
nltk.download('stopwords', quiet=True)

print("Cargando datos...")
# Usar la misma muestra que en el script mejorado para una comparación justa
data = pd.read_csv('Sentiment(1) 1.csv')
data = data[['text','sentiment']]

# Filtrar datos nulos
data = data.dropna(subset=['text', 'sentiment'])

# Usar la misma muestra que en el script mejorado para una comparación justa
data = data.groupby('sentiment').apply(lambda x: x.sample(min(len(x), 1000), random_state=42)).reset_index(drop=True)

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

def wordcloud_draw(words, color='black', filename=None):
    words = ' '.join(words)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=800,
                      height=600
                     ).generate(cleaned_word)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud)
    plt.axis('off')
    
    # Save the figure to a file instead of showing it
    if filename:
        plt.savefig(filename)
        print(f'Wordcloud saved as {filename}')
    plt.close()

print("Generando nubes de palabras...")
print("Palabras positivas")
wordcloud_draw(train_pos, 'white', 'positive_wordcloud_original.png')
print("Palabras negativas")
wordcloud_draw(train_neg, 'black', 'negative_wordcloud_original.png')

print("Preparando datos para clasificación...")
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

w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features

wordcloud_draw(w_features, 'gray', 'features_wordcloud_original.png')

print("Entrenando clasificador Naive Bayes (Original)...")
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Características más informativas:")
print(classifier.most_informative_features(10))

# Evaluar en datos negativos
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
plt.figure(figsize=(8, 5))
plt.bar(['Precisión general', 'Precisión negativos', 'Precisión positivos'], 
        [overall_accuracy, neg_accuracy, pos_accuracy], 
        color=['blue', 'red', 'green'])
plt.title('Precisión del clasificador original')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('original_classifier_metrics.png')
print("Gráfico de métricas guardado como 'original_classifier_metrics.png'")

print("\nProceso completado.")
