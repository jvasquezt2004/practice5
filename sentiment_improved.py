

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


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


print("Cargando datos...")
data = pd.read_csv('Sentiment(1) 1.csv')
data = data[['text', 'sentiment']]


data = data.dropna(subset=['text', 'sentiment'])


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
    
    
    tokens = word_tokenize(text)
    
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) >= 3]
    
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)


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
    
    word_features = list(wordlist.keys())[:2000]  
    return word_features

w_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    
    
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    
    
    if len(document) > 1:
        bigrams = list(nltk.bigrams(document))
        for i in range(len(bigrams)):
            if i < 100:  
                bigram = ' '.join(bigrams[i])
                features['bigram(%s)' % bigram] = True
    
    
    features['length'] = len(document)
    features['has_exclamation'] = '!' in ' '.join(document)
    features['has_question'] = '?' in ' '.join(document)
    
    return features


wordcloud_draw(w_features, 'gray', 'features_wordcloud_improved.png')

print("Preparando datos para clasificación...")

training_set = nltk.classify.apply_features(extract_features, tweets)


print("\nEntrenando clasificador Naive Bayes (Original)...")
classifier_nb = nltk.NaiveBayesClassifier.train(training_set)
print("Características más informativas:")
print(classifier_nb.most_informative_features(10))


print("\nProbando diferentes clasificadores...")


test_pos = test[test['sentiment'] == 'Positive']
test_pos_text = test_pos['processed_text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg_text = test_neg['processed_text']


def evaluate_classifier(classifier, name):
    
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
    
    print(f"\nResultados del clasificador {name}:")
    print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_accuracy:.2%})")
    print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_accuracy:.2%})")
    print(f"Precisión general: {overall_accuracy:.2%}")
    
    return overall_accuracy


nb_accuracy = evaluate_classifier(classifier_nb, "Naive Bayes (Original)")


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


print("\nCombinando clasificadores mediante votación...")
def ensemble_classify(features):
    votes = []
    votes.append(classifier_nb.classify(features))
    votes.append(classifier_mnb.classify(features))
    votes.append(classifier_svc.classify(features))
    votes.append(classifier_rf.classify(features))
    
    
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

print("\nResultados del clasificador Ensemble:")
print(f"[Negative]: {neg_correct}/{len(test_neg_text)} ({neg_correct/len(test_neg_text):.2%})")
print(f"[Positive]: {pos_correct}/{len(test_pos_text)} ({pos_correct/len(test_pos_text):.2%})")
print(f"Precisión general: {ensemble_accuracy:.2%}")


print("\n--- Comparación de Clasificadores ---")
classifiers = ["Naive Bayes (Original)", "MultinomialNB", "LinearSVC", "RandomForest", "Ensemble"]
accuracies = [nb_accuracy, mnb_accuracy, svc_accuracy, rf_accuracy, ensemble_accuracy]


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
