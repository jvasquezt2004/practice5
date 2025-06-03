"""
Script para preprocesar el dataset Sentiment(1) 1.csv.
Realiza limpieza de texto, balanceo de clases, y extracción de características.
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.utils import resample
import string
import datetime

# Descarga de recursos necesarios de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Realiza limpieza y normalización del texto:
    - Convierte a minúsculas
    - Elimina URLs, menciones, hashtags, RT
    - Elimina signos de puntuación
    - Elimina números
    - Elimina espacios múltiples
    - Tokeniza
    - Elimina stopwords
    - Lematiza
    """
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Eliminar menciones, hashtags y RT
    text = re.sub(r'@\w+|#\w+|rt', '', text)
    
    # Eliminar signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenizar
    tokens = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) >= 3]
    
    # Lematizar
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def add_text_features(df):
    """
    Añade características basadas en el texto:
    - Longitud del texto
    - Número de palabras
    - Número de hashtags
    - Número de menciones
    - Contiene URL
    """
    # Preservar el texto original
    df['original_text'] = df['text']
    
    # Características del texto original
    df['text_length'] = df['original_text'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
    df['word_count'] = df['original_text'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
    df['hashtag_count'] = df['original_text'].apply(lambda x: len(re.findall(r'#\w+', str(x))) if isinstance(x, str) else 0)
    df['mention_count'] = df['original_text'].apply(lambda x: len(re.findall(r'@\w+', str(x))) if isinstance(x, str) else 0)
    df['contains_url'] = df['original_text'].apply(lambda x: 1 if re.search(r'http\S+|www\S+', str(x)) else 0)
    df['is_retweet'] = df['original_text'].apply(lambda x: 1 if str(x).lower().startswith('rt @') else 0)
    
    # Aplicar preprocesamiento al texto
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['processed_length'] = df['processed_text'].apply(len)
    df['processed_word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
    
    return df

def balance_classes(df, column='sentiment', method='undersample', random_state=42):
    """
    Balancea las clases en el dataset utilizando diferentes métodos:
    - undersample: reduce las clases mayoritarias
    - oversample: aumenta las clases minoritarias
    - hybrid: combina ambos enfoques
    """
    print(f"Distribución original: {df[column].value_counts().to_dict()}")
    
    # Agrupar por la columna especificada
    groups = {}
    for label, group in df.groupby(column):
        groups[label] = group
    
    # Determinar el tamaño mínimo y máximo
    min_size = min(len(group) for group in groups.values())
    max_size = max(len(group) for group in groups.values())
    
    # Determinar el tamaño objetivo según el método
    if method == 'undersample':
        target_size = min_size
    elif method == 'oversample':
        target_size = max_size
    elif method == 'hybrid':
        target_size = int((min_size + max_size) / 2)
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    # Aplicar el balanceo
    balanced_groups = []
    for label, group in groups.items():
        if len(group) < target_size:
            # Sobremuestreo para clases minoritarias
            balanced_group = resample(group, 
                                     replace=True,
                                     n_samples=target_size,
                                     random_state=random_state)
        elif len(group) > target_size:
            # Submuestreo para clases mayoritarias
            balanced_group = resample(group, 
                                     replace=False,
                                     n_samples=target_size,
                                     random_state=random_state)
        else:
            balanced_group = group
        
        balanced_groups.append(balanced_group)
    
    # Combinar los grupos balanceados
    result = pd.concat(balanced_groups)
    
    print(f"Distribución balanceada: {result[column].value_counts().to_dict()}")
    return result

def add_temporal_features(df):
    """
    Extrae características temporales de la columna 'tweet_created'
    """
    if 'tweet_created' not in df.columns:
        return df
    
    # Convertir a datetime
    df['tweet_datetime'] = pd.to_datetime(df['tweet_created'], errors='coerce')
    
    # Extraer características temporales
    df['tweet_hour'] = df['tweet_datetime'].dt.hour
    df['tweet_day'] = df['tweet_datetime'].dt.day
    df['tweet_month'] = df['tweet_datetime'].dt.month
    df['tweet_year'] = df['tweet_datetime'].dt.year
    df['tweet_dayofweek'] = df['tweet_datetime'].dt.dayofweek
    df['tweet_is_weekend'] = df['tweet_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def oversample_only(df, column='sentiment', random_state=42):
    """
    Balancea las clases en el dataset utilizando SOLO oversampling
    (duplica entradas de clases minoritarias sin reducir ninguna clase)
    """
    print(f"Distribución original: {df[column].value_counts().to_dict()}")
    
    # Agrupar por la columna especificada
    groups = {}
    for label, group in df.groupby(column):
        groups[label] = group
    
    # Determinar el tamaño máximo
    max_size = max(len(group) for group in groups.values())
    
    # Aplicar el oversampling solo a clases minoritarias
    balanced_groups = []
    for label, group in groups.items():
        if len(group) < max_size:
            # Sobremuestreo para clases minoritarias
            balanced_group = resample(group, 
                                     replace=True,
                                     n_samples=max_size,
                                     random_state=random_state)
        else:
            # Mantener clases mayoritarias sin cambios
            balanced_group = group
        
        balanced_groups.append(balanced_group)
    
    # Combinar los grupos balanceados
    result = pd.concat(balanced_groups)
    
    print(f"Distribución balanceada (oversampling): {result[column].value_counts().to_dict()}")
    return result

def main():
    # Cargar el dataset original
    input_file = "/home/alonso/Code/School/NLP/practice5/Sentiment(1) 1.csv"
    print(f"Cargando dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Información del dataset original
    print(f"Dimensiones originales: {df.shape}")
    
    # Eliminar filas con valores nulos en columnas importantes
    print("Manejando valores nulos...")
    df_clean = df.dropna(subset=['text', 'sentiment'])
    print(f"Filas después de eliminar nulos: {df_clean.shape[0]}")
    
    # Añadir características de texto
    print("Añadiendo características de texto...")
    df_features = add_text_features(df_clean)
    
    # Añadir características temporales
    print("Añadiendo características temporales...")
    df_features = add_temporal_features(df_features)
    
    # Crear diferentes versiones del dataset
    
    # 1. Dataset completo con todas las características adicionales
    output_file_full = "/home/alonso/Code/School/NLP/practice5/sentiment_processed_full.csv"
    df_features.to_csv(output_file_full, index=False)
    print(f"Dataset completo guardado: {output_file_full}")
    
    # 2. Dataset balanceado (solo características esenciales)
    print("Creando dataset balanceado...")
    columns_to_keep = ['id', 'candidate', 'sentiment', 'subject_matter', 
                       'original_text', 'processed_text', 
                       'text_length', 'word_count', 'hashtag_count', 'mention_count', 
                       'contains_url', 'is_retweet']
    
    df_essential = df_features[columns_to_keep].copy()
    
    # Balance usando método híbrido (mejor equilibrio entre clases)
    df_balanced = balance_classes(df_essential, method='hybrid')
    
    output_file_balanced = "/home/alonso/Code/School/NLP/practice5/sentiment_processed_balanced.csv"
    df_balanced.to_csv(output_file_balanced, index=False)
    print(f"Dataset balanceado guardado: {output_file_balanced}")
    
    # 3. Dataset para análisis binario (solo positivo/negativo)
    print("Creando dataset binario (positivo/negativo)...")
    df_binary = df_features[df_features['sentiment'] != 'Neutral'].copy()
    
    # Balance de clases binarias
    df_binary_balanced = balance_classes(df_binary, method='undersample')
    
    output_file_binary = "/home/alonso/Code/School/NLP/practice5/sentiment_processed_binary.csv"
    df_binary_balanced.to_csv(output_file_binary, index=False)
    print(f"Dataset binario guardado: {output_file_binary}")
    
    # 4. NUEVO: Dataset binario balanceado solo con oversampling
    print("Creando dataset binario con OVERSAMPLING (sin reducir clases)...")
    df_binary_oversampled = df_features[df_features['sentiment'] != 'Neutral'].copy()
    
    # Balance usando solo oversampling (sin reducir ninguna clase)
    df_binary_oversampled = oversample_only(df_binary_oversampled)
    
    output_file_oversampled = "/home/alonso/Code/School/NLP/practice5/sentiment_processed_oversampled.csv"
    df_binary_oversampled.to_csv(output_file_oversampled, index=False)
    print(f"Dataset con oversampling guardado: {output_file_oversampled}")
    
    # Visualizar la distribución de sentimientos en los diferentes datasets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dataset original
    df_clean['sentiment'].value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Original')
    axes[0, 0].set_ylabel('Frecuencia')
    
    # Dataset balanceado
    df_balanced['sentiment'].value_counts().plot(kind='bar', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Balanceado (Hybrid)')
    
    # Dataset binario
    df_binary_balanced['sentiment'].value_counts().plot(kind='bar', ax=axes[1, 0], color='salmon')
    axes[1, 0].set_title('Binario (Pos/Neg)')
    
    # Dataset con oversampling
    df_binary_oversampled['sentiment'].value_counts().plot(kind='bar', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Oversampling (sin reducción)')
    
    plt.tight_layout()
    plt.savefig('/home/alonso/Code/School/NLP/practice5/sentiment_distributions.png')
    print("Gráfico de distribuciones guardado como 'sentiment_distributions.png'")
    
    print("Preprocesamiento completado exitosamente.")

if __name__ == "__main__":
    main()
