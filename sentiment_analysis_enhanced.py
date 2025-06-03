# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import random
import io
import re

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


def get_sentences_from_datasets(num_regular=50, num_tricky=50):
    regular_sentences = []
    tricky_sentences = []
    

    imdb_url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    amazon_reviews_url = "https://raw.githubusercontent.com/qdata/deep-learning-for-sequence/master/data/amazon/amazon_reviews_small.csv"
    stanford_url = "https://raw.githubusercontent.com/suraj-deshmukh/Movie-Review-Sentiment-Analysis/master/test_pos.txt"
    

    try:
        print("Downloading regular sentences from IMDB dataset...")
        response = requests.get(imdb_url, timeout=10)
        response.raise_for_status()
        

        try:
            data = pd.read_csv(io.StringIO(response.text), usecols=['review', 'sentiment'])
            

            data['length'] = data['review'].apply(len)
            short_reviews = data[data['length'] < 100]
            

            positive_reviews = short_reviews[short_reviews['sentiment'] == 'positive']['review'].tolist()
            negative_reviews = short_reviews[short_reviews['sentiment'] == 'negative']['review'].tolist()
            

            if len(positive_reviews) < num_regular//2 or len(negative_reviews) < num_regular//2:
                positive_reviews = data[data['sentiment'] == 'positive']['review'].tolist()
                negative_reviews = data[data['sentiment'] == 'negative']['review'].tolist()
            

            num_each = num_regular // 2
            pos_samples = random.sample(positive_reviews, min(num_each, len(positive_reviews)))
            neg_samples = random.sample(negative_reviews, min(num_each, len(negative_reviews)))
            
        except Exception as e:

            print(f"Error processing IMDB CSV: {e}")

            lines = response.text.split('\n')[:1000]  # Limitar a primeras 1000 líneas
            pos_reviews = []
            neg_reviews = []
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split(',', 1)  # Dividir en máximo 2 partes
                if len(parts) < 2:
                    continue
                    
                review = parts[0].strip('"\'').strip()
                sentiment = parts[-1].strip('"\'').strip()
                

                if 'positive' in sentiment.lower() or 'pos' in sentiment.lower():
                    pos_reviews.append(review)
                elif 'negative' in sentiment.lower() or 'neg' in sentiment.lower():
                    neg_reviews.append(review)
            
            # Seleccionar muestras
            num_each = num_regular // 2
            pos_samples = random.sample(pos_reviews, min(num_each, len(pos_reviews))) if pos_reviews else []
            neg_samples = random.sample(neg_reviews, min(num_each, len(neg_reviews))) if neg_reviews else []
        
        regular_sentences.extend(pos_samples)
        regular_sentences.extend(neg_samples)
        
    except Exception as e:
        print(f"Error downloading Twitter dataset: {e}")
        print("Using fallback regular sentences...")
        
        fallback_regular = [
            "I absolutely love this product, it's amazing!",
            "The service was excellent and the staff was very friendly.",
            "This is the best movie I've seen all year!",
            "I had a wonderful experience at the restaurant last night.",
            "The book was very well written and engaging from start to finish.",
            "This is the worst service I've ever experienced.",
            "I would never recommend this product to anyone.",
            "The movie was terrible and a complete waste of time.",
            "I'm extremely disappointed with the quality of this item.",
            "The customer support was unhelpful and rude.",
            "I love this movie.",
            "The service was excellent.",
            "She is very talented.",
            "The weather is beautiful today.",
            "The concert was amazing.",
            "This product exceeded my expectations.",
            "The view from the hotel was breathtaking.",
            "The staff were incredibly helpful.",
            "This book is a masterpiece.",
            "The food at the restaurant was delicious.",
            "Their customer service is outstanding.",
            "The performance was brilliant.",
            "I highly recommend this place.",
            "The graphics in this game are stunning.",
            "The meeting was productive.",
            "This software is very intuitive.",
            "The design is elegant and modern.",
            "The app is user-friendly.",
            "The conference was informative.",
            "This course has been very valuable.",
            "Their presentation was impressive.",
            "The coffee here is the best in town.",
            "Their support team resolved my issue quickly.",
            "The upgrade was worth every penny.",
            "This camera takes beautiful photos.",
            "This film is terrible.",
            "The food was disgusting.",
            "He performed poorly.",
            "This book is boring.",
            "That was a waste of money.",
            "The service at this restaurant was horrible.",
            "I regret buying this product.",
            "Their website is confusing and difficult to navigate.",
            "The movie plot made no sense.",
            "This laptop battery drains too quickly.",
            "The hotel room was dirty and uncomfortable.",
            "Their response time was unacceptably slow.",
            "This app keeps crashing on my phone.",
            "The sound quality is very poor.",
            "The instructions were unclear.",
            "This course was a complete disappointment.",
            "Their customer service is non-existent.",
            "The concert venue was overcrowded.",
            "This keyboard is uncomfortable to type on.",
            "The flight was delayed for hours.",
            "Their pricing is completely unreasonable.",
            "The meeting was a waste of time.",
            "This coffee tastes like dishwater.",
            "The interface is outdated and clunky.",
            "Their refund policy is terrible."
        ]
        
        regular_sentences.extend(fallback_regular[:num_regular])
    

    try:

        print("Downloading tricky sentences from Amazon Reviews & Stanford datasets...")
        amazon_response = requests.get(amazon_reviews_url, timeout=10)
        amazon_response.raise_for_status()
        

        try:

            amazon_data = pd.read_csv(io.StringIO(amazon_response.text))
            

            if 'reviewText' in amazon_data.columns:
                amazon_text_col = 'reviewText'
            elif 'review' in amazon_data.columns:
                amazon_text_col = 'review'
            elif 'text' in amazon_data.columns:
                amazon_text_col = 'text'
            else:

                amazon_text_col = amazon_data.columns[0]
            

            amazon_reviews = amazon_data[amazon_text_col].tolist()
            
        except Exception as e:
            print(f"Error processing Amazon CSV: {e}")

            amazon_reviews = []
            lines = amazon_response.text.split('\n')[:500]  # Limitar a primeras 500 líneas
            
            for line in lines:
                if not line.strip() or line.startswith('#'):
                    continue
                    

                parts = line.split(',', 3)  # Dividir en máximo 4 partes
                if len(parts) >= 3:
                    review = parts[2].strip('"\'').strip()
                    if len(review) > 20:  # Solo considerar reseñas no vacías
                        amazon_reviews.append(review)
        

        stanford_response = requests.get(stanford_url, timeout=10)
        stanford_response.raise_for_status()
        

        stanford_reviews = [line.strip() for line in stanford_response.text.split('\n') if line.strip()]
        

        combined_reviews = amazon_reviews + stanford_reviews
        

        
        tricky_candidates = []
        contrast_words = ['but', 'however', 'although', 'yet', 'though', 'while', 'nevertheless', 'despite']
        negation_words = ['not', "n't", 'never', 'no', 'nothing', 'neither', 'nor']
        
        for review in combined_reviews:

            review_str = str(review).lower()

            word_count = len(review_str.split())
            

            if word_count > 10:
                has_contrast = any(word in review_str for word in contrast_words)
                has_negation = any(word in review_str for word in negation_words)
                
                if has_contrast or has_negation:
                    tricky_candidates.append(review)
        

        if len(tricky_candidates) < num_tricky:

            long_reviews = [r for r in combined_reviews if len(str(r).split()) > 15 
                          and r not in tricky_candidates]
            tricky_candidates.extend(long_reviews)
        
        # Seleccionar muestras aleatorias
        if tricky_candidates:

            if len(tricky_candidates) >= num_tricky:

                contrast_candidates = [s for s in tricky_candidates if any(word in str(s).lower() for word in contrast_words)]
                negation_candidates = [s for s in tricky_candidates if any(word in str(s).lower() for word in negation_words)]
                long_candidates = [s for s in tricky_candidates if len(str(s).split()) > 20 and 
                                 s not in contrast_candidates and s not in negation_candidates]
                other_candidates = [s for s in tricky_candidates if s not in contrast_candidates and 
                                  s not in negation_candidates and s not in long_candidates]
                

                num_contrast = min(num_tricky // 3, len(contrast_candidates))
                num_negation = min(num_tricky // 3, len(negation_candidates))
                num_long = min(num_tricky // 6, len(long_candidates))
                num_other = num_tricky - (num_contrast + num_negation + num_long)
                

                selected_tricky = []
                if contrast_candidates and num_contrast > 0:
                    selected_tricky.extend(random.sample(contrast_candidates, num_contrast))
                if negation_candidates and num_negation > 0:
                    selected_tricky.extend(random.sample(negation_candidates, num_negation))
                if long_candidates and num_long > 0:
                    selected_tricky.extend(random.sample(long_candidates, num_long))
                

                remaining_candidates = [s for s in tricky_candidates if s not in selected_tricky]
                if remaining_candidates and num_other > 0:
                    needed = min(num_other, len(remaining_candidates))
                    selected_tricky.extend(random.sample(remaining_candidates, needed))
                

                if len(selected_tricky) < num_tricky:
                    remaining_needed = num_tricky - len(selected_tricky)
                    print(f"Still need {remaining_needed} more tricky sentences, adding from fallback...")
                    selected_tricky.extend(fallback_tricky[:remaining_needed])
                    
                tricky_sentences.extend(selected_tricky)
            else:

                tricky_sentences.extend(tricky_candidates)

                print("Not enough tricky sentences, adding fallback tricky examples...")
                remaining = num_tricky - len(tricky_candidates)
                

                contrast_fallback = fallback_tricky[:20]
                negation_fallback = fallback_tricky[20:30]
                long_fallback = fallback_tricky[30:40]
                sarcasm_fallback = fallback_tricky[40:]
                

                num_contrast_fallback = remaining // 4
                num_negation_fallback = remaining // 4
                num_long_fallback = remaining // 4
                num_sarcasm_fallback = remaining - (num_contrast_fallback + num_negation_fallback + num_long_fallback)
                
                fallback_selected = []
                fallback_selected.extend(random.sample(contrast_fallback, min(num_contrast_fallback, len(contrast_fallback))))
                fallback_selected.extend(random.sample(negation_fallback, min(num_negation_fallback, len(negation_fallback))))
                fallback_selected.extend(random.sample(long_fallback, min(num_long_fallback, len(long_fallback))))
                fallback_selected.extend(random.sample(sarcasm_fallback, min(num_sarcasm_fallback, len(sarcasm_fallback))))
                

                if len(fallback_selected) < remaining:
                    still_needed = remaining - len(fallback_selected)
                    remaining_fallback = [f for f in fallback_tricky if f not in fallback_selected]
                    if remaining_fallback:
                        fallback_selected.extend(random.sample(remaining_fallback, min(still_needed, len(remaining_fallback))))
                
                tricky_sentences.extend(fallback_selected)
        
    except Exception as e:
        print(f"Error downloading Amazon/Yelp datasets: {e}")
        print("Using fallback tricky sentences...")
    

    if len(tricky_sentences) < num_tricky:
        print("Not enough tricky sentences, adding fallback tricky examples...")
        
        fallback_tricky = [

            "The food was amazing but the service was terrible.",
            "I don't hate this product, but I wouldn't recommend it either.",
            "It's not the worst movie I've seen, but definitely not the best either.",
            "While I enjoyed the book overall, I found the ending disappointing.",
            "The product works exactly as advertised, however it's too expensive to be worth it.",
            "Although the hotel had a beautiful view, the room itself was quite dirty.",
            "The staff were friendly but seemed inexperienced and made several mistakes.",
            "I can't say I disliked the meal, yet I probably won't order it again.",
            "The movie wasn't as bad as the critics said, though it could have been better.",
            "Despite having great features, the app crashes too often to be useful.",
            "The concert started off amazing, however the sound quality deteriorated.",
            "It's not that I don't appreciate the effort, but the results are mediocre.",
            "The phone has excellent battery life but the camera quality is disappointing.",
            "Although I initially liked the design, it's actually not very practical.",
            "The software is powerful, yet the learning curve is unnecessarily steep.",
            "I love the plot of this movie, but the acting was terrible.",
            "The scenery was beautiful, however the tour guide was boring.",
            "Despite the high price, the quality doesn't match expectations.",
            "While the first half was engaging, the second half dragged on too long.",
            "The concept is innovative, but the execution leaves much to be desired.",
            

            "I wouldn't say it was a bad experience, just not what I expected.",
            "This isn't the worst restaurant in town, but I've had much better meals elsewhere.",
            "The service wasn't terrible, yet it certainly wasn't impressive either.",
            "I don't think the movie deserved all the negative reviews it received.",
            "The product doesn't fail completely, it just doesn't excel in any particular area.",
            "This isn't exactly what I was looking for, though it might work for others.",
            "The hotel wasn't as luxurious as advertised, but it wasn't uncomfortable.",
            "I wouldn't call the staff rude, but they definitely weren't friendly or helpful.",
            "The app doesn't crash often, but when it does, you lose all your progress.",
            "The food wasn't inedible, just bland and overpriced for what you get.",
            

            "After considering all the positive and negative aspects of the experience, I'm still not entirely sure whether I'd recommend it to others or not.",
            "On one hand the interface is intuitive and easy to navigate, but on the other hand the functionality is limited compared to similar products on the market.",
            "What started as an excellent novel with captivating characters and an intriguing plot gradually devolved into a predictable story with an unsatisfying conclusion.",
            "The restaurant creates a wonderful ambiance with its decor and music, yet the menu is overpriced for the quality and quantity of food you receive.",
            "While the customer service team was quick to respond to my inquiry, they were unable to provide a solution to my problem and seemed more interested in closing the ticket than helping me.",
            "The hotel offers spectacular views and luxurious amenities that justify its premium price, though the location is inconvenient for anyone wanting to explore the main attractions of the city.",
            "For the price point, I expected better performance from this device, although I must admit that the design is sleek and it has some innovative features not found in competing products.",
            "The course material was comprehensive and well-structured, however the instructor's teaching style was monotonous and failed to keep students engaged during the lengthy sessions.",
            "Initially I was impressed by the product's quality and performance, but after a few months of use, several issues emerged that make me question its long-term durability and value.",
            "The first half of the movie brilliantly builds tension and develops characters you care about, only to squander that potential with a rushed third act that leaves too many questions unanswered.",
            

            "Oh sure, waiting two hours for a table at a restaurant that serves mediocre food at premium prices is exactly my idea of a perfect evening.",
            "The customer service was so helpful that I ended up solving the problem myself after being on hold for 45 minutes.",
            "If you enjoy products that work perfectly until just after the warranty expires, then this is definitely the one for you.",
            "The movie was so original that I could predict every plot twist within the first ten minutes.",
            "The 'luxury' hotel room was exactly as advertised, if your definition of luxury includes stained carpets and noisy air conditioning.",
            "The 'fast' shipping option was amazing - my package arrived only two weeks after the estimated delivery date.",
            "The concert was a real bargain considering I paid premium prices to barely see the stage and listen to terrible acoustics.",
            "The software is brilliantly designed to test your patience with constant updates that add bugs rather than fixing them.",
            "The restaurant portion sizes are perfect if you're on a diet and don't actually want to feel full after paying for a complete meal.",
            "The instruction manual was incredibly helpful, assuming you already know exactly how to use the product."
        ]
        

        needed = num_tricky - len(tricky_sentences)
        tricky_sentences.extend(fallback_tricky[:needed])
    

    clean_regular = [re.sub(r'http\S+|www\S+|RT |\s+', ' ', str(s)).strip() for s in regular_sentences]
    clean_tricky = [re.sub(r'http\S+|www\S+|RT |\s+', ' ', str(s)).strip() for s in tricky_sentences]
    
    return clean_regular[:num_regular], clean_tricky[:num_tricky]


import os
import argparse
import sys


from subprocess import check_output
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


def main():

    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    print(f"Downloading required NLTK resources to {nltk_data_dir}...")

    nltk.download('subjectivity', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('vader_lexicon', download_dir=nltk_data_dir)
    
    parser = argparse.ArgumentParser(description='Sentiment analysis with NLTK')
    parser.add_argument('--n_instances', type=int, default=1000, help='Number of instances to load')
    parser.add_argument('--version', type=str, default='enhanced', help='Version identifier for output file')
    args = parser.parse_args()

    n_instances = args.n_instances

    visualize_stopwords()

    output_filename = f'output_{args.version}.txt'
    original_stdout = sys.stdout
    f = open(output_filename, 'w')
    sys.stdout = f

    print(f"Sentiment Analysis with NLTK - Enhanced Version")
    

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
    
    print("Naive Bayes Classifier Evaluation:")
    for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))



    num_sentences = 50  # Cantidad de frases a obtener de cada tipo
    regular_sentences, new_tricky_sentences = get_sentences_from_datasets(num_sentences, num_sentences)
    
    print(f"\nAdded {len(regular_sentences)} new regular sentences and {len(new_tricky_sentences)} new tricky sentences from datasets")
    

    original_sentences = [
        "VADER is smart, handsome, and funny.",
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


    original_tricky_sentences = [
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


    sentences = original_sentences + regular_sentences
    tricky_sentences = original_tricky_sentences + new_tricky_sentences
    
    print("\nTotal sentences for analysis:", len(sentences))
    sentences.extend(tricky_sentences)


    stop_words = set(stopwords.words('english'))
    sentences = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in sentences]


    paragraph = "It was one of the worst movies I've seen, despite good reviews. \
     Unbelievably bad acting!! Poor direction. VERY poor production. \
     The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

    paragraph = ' '.join([word for word in paragraph.split() if word not in stop_words])
    lines_list = tokenize.sent_tokenize(paragraph)
    sentences.extend(lines_list)


    print("\nVADER Sentiment Analysis on Sample Sentences:")
    sid = SentimentIntensityAnalyzer()
    for sentence in sentences:
         print(sentence)
         ss = sid.polarity_scores(sentence)
         for k in sorted(ss):
             print('{0}: {1}, '.format(k, ss[k]), end='')
         print()


    print("\nSVM Classifier results:")
    svm_clf = SklearnClassifier(SVC())
    classifier_svm = sentim_analyzer.train(svm_clf.train, training_set)
    for key, value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))

    sys.stdout = original_stdout
    f.close()
    print(f"Analysis complete. Results saved to {output_filename}")


if __name__ == "__main__":
    main()
