

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


nltk.download('stopwords', quiet=True)


print("Cargando datos...")
data = pd.read_csv('Sentiment(1) 1.csv')
data = data[['text', 'sentiment']]


data = data.dropna(subset=['text', 'sentiment'])



data = data.groupby('sentiment').apply(lambda x: x.sample(min(len(x), 1000), random_state=42)).reset_index(drop=True)


train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])


train = train.query("sentiment != 'Neutral'")
test = test.query("sentiment != 'Neutral'")

print(f"Datos de entrenamiento: {train.shape[0]} tweets")
print(f"Datos de prueba: {test.shape[0]} tweets")


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    
    text = text.lower()
    
    
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+|rt', '', text)
    
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    text = re.sub(r'\d+', '', text)
    
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    
    words = [w for w in text.split() if len(w) >= 3 and w not in stopwords.words('english')]
    
    return ' '.join(words)


print("Preprocesando textos...")
train['processed_text'] = train['text'].apply(preprocess_text)
test['processed_text'] = test['text'].apply(preprocess_text)


train_pos = train[train['sentiment'] == 'Positive']
train_pos_text = train_pos['processed_text']
train_neg = train[train['sentiment'] == 'Negative']
train_neg_text = train_neg['processed_text']


def wordcloud_draw(words, color='black', filename=None):
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
        print(f'Wordcloud saved as {filename}')
    plt.close()

print("Generando nubes de palabras...")
print("Palabras positivas")
wordcloud_draw(train_pos_text, 'white', 'positive_wordcloud_light.png')
print("Palabras negativas")
wordcloud_draw(train_neg_text, 'black', 'negative_wordcloud_light.png')


tweets = []
for index, row in train.iterrows():
    words = row.processed_text.split()
    tweets.append((words, row.sentiment))


def get_words_in_tweets(tweets):
    all_words = []
    for (words, _) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    
    word_features = list(wordlist.keys())[:500]
    return word_features

print("Extrayendo características...")
w_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    
    
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    
    
    features['length'] = len(document)
    
    return features


wordcloud_draw(w_features, 'gray', 'features_wordcloud_light.png')

print("Preparando datos para clasificación...")

training_set = nltk.classify.apply_features(extract_features, tweets)


print("\nEntrenando clasificador Naive Bayes (Original)...")
classifier_nb = nltk.NaiveBayesClassifier.train(training_set)
print("Características más informativas:")
print(classifier_nb.most_informative_features(10))


print("\nEntrenando clasificador MultinomialNB...")
classifier_mnb = SklearnClassifier(MultinomialNB())
classifier_mnb.train(training_set)


test_pos = test[test['sentiment'] == 'Positive']
test_pos_text = test_pos['processed_text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg_text = test_neg['processed_text']


def evaluate_classifier(classifier, name):
    print(f"\nEvaluando clasificador {name}...")
    
    
    neg_correct = 0
    for obj in test_neg_text:
        words = obj.split()
        if classifier.classify(extract_features(words)) == 'Negative':
            neg_correct += 1
    
    
    pos_correct = 0
    for obj in test_pos_text:
        words = obj.split()
        if classifier.classify(extract_features(words)) == 'Positive':
            pos_correct += 1
    
    
    neg_accuracy = neg_correct / len(test_neg_text) if len(test_neg_text) > 0 else 0
    pos_accuracy = pos_correct / len(test_pos_text) if len(test_pos_text) > 0 else 0
    overall_accuracy = (neg_correct + pos_correct) / (len(test_neg_text) + len(test_pos_text))
    
    print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_accuracy:.2%})")
    print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_accuracy:.2%})")
    print(f"Precisión general: {overall_accuracy:.2%}")
    
    return overall_accuracy


print("\n--- Evaluación de Clasificadores ---")
nb_accuracy = evaluate_classifier(classifier_nb, "Naive Bayes (Original)")
mnb_accuracy = evaluate_classifier(classifier_mnb, "MultinomialNB")


print("\n--- Clasificador Ensemble Simplificado ---")
def ensemble_classify(features):
    votes = []
    votes.append(classifier_nb.classify(features))
    votes.append(classifier_mnb.classify(features))
    
    
    if votes.count('Positive') > votes.count('Negative'):
        return 'Positive'
    else:
        return 'Negative'


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


print("\n--- Comparación de Clasificadores ---")
classifiers = ["Naive Bayes", "MultinomialNB", "Ensemble"]
accuracies = [nb_accuracy, mnb_accuracy, ensemble_accuracy]


plt.figure(figsize=(8, 5))
plt.bar(classifiers, accuracies, color=['blue', 'green', 'orange'])
plt.title('Comparación de Precisión entre Clasificadores')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('classifier_comparison_light.png')
print("Gráfico de comparación guardado como 'classifier_comparison_light.png'")


print("\n--- Resumen de Mejoras Implementadas ---")
print("1. Preprocesamiento mejorado del texto")
print("2. Selección de características más relevantes")
print("3. Características adicionales simples (longitud)")
print("4. Uso de clasificador MultinomialNB de scikit-learn")
print("5. Votación entre clasificadores (ensemble)")

print("\nProceso completado.")
