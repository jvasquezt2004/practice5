# -*- coding: utf-8 -*-




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from wordcloud import WordCloud

import matplotlib.pyplot as plt


def visualize_stopwords():
    nltk.download('stopwords', quiet=True)
    stop_words = stopwords.words('english')
    
    stopwords_str = ' '.join(stop_words)
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        max_words=100,
        contour_width=3, 
        contour_color='steelblue'
    ).generate(stopwords_str)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('NLTK Stopwords Visualization')
    plt.tight_layout()
    
    plt.savefig('stopwords_wordcloud.png')
    print('Stopwords wordcloud saved as stopwords_wordcloud.png')
    plt.close()


import os
import argparse
import sys


from subprocess import check_output
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


parser = argparse.ArgumentParser(description='Sentiment analysis with NLTK')
parser.add_argument('--n_instances', type=int, default=1000, help='Number of instances to load')
parser.add_argument('--version', type=str, default='1', help='Version identifier for output file')
args = parser.parse_args()

n_instances = args.n_instances

visualize_stopwords()

output_filename = f'output_{args.version}.txt'
original_stdout = sys.stdout
f = open(output_filename, 'w')
sys.stdout = f

subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

train_subj_docs, test_subj_docs = train_test_split(subj_docs, test_size=0.2, random_state=42)
train_obj_docs, test_obj_docs = train_test_split(obj_docs, test_size=0.2, random_state=42)
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))



sentences = ["VADER is smart, handsome, and funny.",
   "VADER is smart, handsome, and funny!",
   "VADER is very smart, handsome, and funny.",
   "VADER is VERY SMART, handsome, and FUNNY.",
   "VADER is VERY SMART, handsome, and FUNNY!!!",
   "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",
   "The book was good.",
   "The book was kind of good.",
   "The plot was good, but the characters are uncompelling and the dialog is not great.",
   "A really bad, horrible book.",
   "At least it isn't a horrible book.",
   ":) and :D",
   "",
   "Today sux",
   "Today sux!",
   "Today SUX!",
   "Today kinda sux! But I'll get by, lol"
]

tricky_sentences = [
    "Most automated sentiment analysis tools are shit.",
    "VADER sentiment analysis is the shit.",
    "Sentiment analysis has never been good.",
    "Sentiment analysis with VADER has never been this good.",
    "Warren Beatty has never been so entertaining.",
    "I won't say that the movie is astounding and I wouldn't claim that \
    the movie is too banal either.",
    "I like to hate Michael Bay films, but I couldn't fault this one",
    "It's one thing to watch an Uwe Boll film, but another thing entirely \
    to pay for it",
    "The movie was too good",
    "This movie was actually neither that funny, nor super witty.",
    "This movie doesn't care about cleverness, wit or any other kind of \
    intelligent humor.",
    "Those who find ugly meanings in beautiful things are corrupt without \
    being charming.",
    "There are slow and repetitive parts, BUT it has just enough spice to \
    keep it interesting.",
    "The script is not fantastic, but the acting is decent and the cinematography \
    is EXCELLENT!",
    "Roger Dodger is one of the most compelling variations on this theme.",
    "Roger Dodger is one of the least compelling variations on this theme.",
    "Roger Dodger is at least compelling as a variation on the theme.",
    "they fall in love with the product",
    "but then it breaks",
    "usually around the time the 90 day warranty expires",
    "the twin towers collapsed today",
    "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
    under orders and in the ''least offensive way possible.''"
 ]

sentences.extend(tricky_sentences)

stop_words = set(stopwords.words('english'))
sentences = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in sentences]

paragraph = "It was one of the worst movies I've seen, despite good reviews. \
 Unbelievably bad acting!! Poor direction. VERY poor production. \
 The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

paragraph = ' '.join([word for word in paragraph.split() if word not in stop_words])
lines_list = tokenize.sent_tokenize(paragraph)
sentences.extend(lines_list)

sid = SentimentIntensityAnalyzer()
for sentence in sentences:
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()

svm_clf = SklearnClassifier(SVC())
classifier_svm = sentim_analyzer.train(svm_clf.train, training_set)
print("SVM Classifier results:")
for key, value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

sys.stdout = original_stdout
f.close()
