import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)

def analyze_sentiment_data(csv_path):

    
    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    
    
    print("\n--- INFORMACIÓN BÁSICA DEL DATASET ---")
    print(f"Número total de tweets: {len(df)}")
    print(f"Columnas disponibles: {', '.join(df.columns)}")
    print("\n--- RESUMEN ESTADÍSTICO ---")
    print(df.describe(include='all'))
    
    
    print("\n--- VALORES NULOS POR COLUMNA ---")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])
    
    
    
    
    print("\n--- BALANCE DE SENTIMIENTOS ---")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    
    
    plt.figure()
    ax = sentiment_counts.plot(kind='bar')
    ax.set_title('Distribución de Sentimientos')
    ax.set_ylabel('Número de tweets')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    
    
    print("\n--- BALANCE DE CANDIDATOS MENCIONADOS ---")
    candidate_counts = df['candidate'].value_counts()
    print(candidate_counts)
    
    
    plt.figure(figsize=(14, 8))
    candidate_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Candidatos Mencionados')
    plt.ylabel('Número de tweets')
    plt.tight_layout()
    plt.savefig('candidate_distribution.png')
    
    
    print("\n--- BALANCE DE TEMAS ---")
    subject_counts = df['subject_matter'].value_counts()
    print(subject_counts)
    
    
    plt.figure(figsize=(14, 8))
    subject_counts.head(10).plot(kind='bar')
    plt.title('Distribución de Temas')
    plt.ylabel('Número de tweets')
    plt.tight_layout()
    plt.savefig('subject_distribution.png')
    
    
    
    
    print("\n--- RELACIÓN ENTRE CANDIDATOS Y SENTIMIENTOS ---")
    candidate_sentiment = pd.crosstab(df['candidate'], df['sentiment'])
    print(candidate_sentiment)
    
    
    plt.figure(figsize=(15, 10))
    top_candidates = candidate_counts.head(8).index
    candidate_sentiment_filtered = candidate_sentiment.loc[top_candidates]
    
    
    candidate_sentiment_norm = candidate_sentiment_filtered.div(candidate_sentiment_filtered.sum(axis=1), axis=0)
    candidate_sentiment_norm.plot(kind='bar', stacked=True)
    plt.title('Proporción de Sentimientos por Candidato')
    plt.ylabel('Proporción')
    plt.tight_layout()
    plt.savefig('candidate_sentiment_relation.png')
    
    
    print("\n--- RELACIÓN ENTRE TEMAS Y SENTIMIENTOS ---")
    subject_sentiment = pd.crosstab(df['subject_matter'], df['sentiment'])
    print(subject_sentiment)
    
    plt.figure(figsize=(15, 10))
    top_subjects = subject_counts.head(8).index
    subject_sentiment_filtered = subject_sentiment.loc[top_subjects]
    
    
    subject_sentiment_norm = subject_sentiment_filtered.div(subject_sentiment_filtered.sum(axis=1), axis=0)
    subject_sentiment_norm.plot(kind='bar', stacked=True)
    plt.title('Proporción de Sentimientos por Tema')
    plt.ylabel('Proporción')
    plt.tight_layout()
    plt.savefig('subject_sentiment_relation.png')
    
    
    if 'tweet_created' in df.columns:
        print("\n--- ANÁLISIS TEMPORAL ---")
        
        df['tweet_date'] = pd.to_datetime(df['tweet_created'])
        df['tweet_date'] = df['tweet_date'].dt.date
        
        
        date_counts = df['tweet_date'].value_counts().sort_index()
        print(date_counts)
        
        plt.figure(figsize=(14, 6))
        date_counts.plot(kind='line', marker='o')
        plt.title('Volumen de Tweets por Día')
        plt.ylabel('Número de tweets')
        plt.tight_layout()
        plt.savefig('temporal_analysis.png')
    
    
    plt.figure()
    sns.histplot(df['sentiment_confidence'], bins=20, kde=True)
    plt.title('Distribución de Confianza en los Sentimientos')
    plt.xlabel('Nivel de Confianza')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('sentiment_confidence_distribution.png')
    
    print("\nAnálisis completado. Se han generado gráficos para visualizar los resultados.")
    
    return df

if __name__ == "__main__":
    
    csv_path = "/home/alonso/Code/School/NLP/practice5/Sentiment(1) 1.csv"
    
    
    df = analyze_sentiment_data(csv_path)
