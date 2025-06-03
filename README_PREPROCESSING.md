# Preprocesamiento del Dataset Sentiment(1)

Este documento detalla el proceso de preprocesamiento aplicado al dataset de sentimientos de tweets (`Sentiment(1) 1.csv`), explicando las técnicas utilizadas y los archivos generados.

## Objetivo del Preprocesamiento

El preprocesamiento tiene como finalidad:

1. Limpiar y normalizar los textos de los tweets
2. Extraer características adicionales útiles para el análisis
3. Corregir el desbalance entre clases de sentimientos
4. Crear versiones alternativas del dataset para diferentes tipos de análisis

## Técnicas de Preprocesamiento de Texto

### Limpieza de Texto

El texto de cada tweet se procesa mediante las siguientes operaciones:

- **Normalización**: Conversión a minúsculas
- **Eliminación de elementos específicos**:
  - URLs (`http://...`, `www....`)
  - Menciones a usuarios (`@usuario`)
  - Hashtags (`#tema`)
  - Indicadores de retweet (`RT`)
- **Eliminación de caracteres especiales**:
  - Signos de puntuación
  - Números
  - Espacios múltiples
- **Tokenización**: Separación del texto en palabras individuales
- **Filtrado**:
  - Eliminación de stopwords (palabras comunes sin valor semántico)
  - Eliminación de palabras muy cortas (menos de 3 caracteres)
- **Lematización**: Reducción de palabras a su forma base o lema

Ejemplo de preprocesamiento:
```
Original: "RT @usuario: ¡El #GOPDebate fue increíble! Trump habló sobre http://example.com 👍"
Procesado: "debate increíble trump habló"
```

## Extracción de Características

Se extrajeron las siguientes características adicionales para enriquecer el análisis:

### Características de Texto

- **Longitud del texto original**: Número de caracteres
- **Número de palabras**: Conteo de palabras en el texto original
- **Número de hashtags**: Conteo de elementos que comienzan con `#`
- **Número de menciones**: Conteo de elementos que comienzan con `@`
- **Presencia de URL**: Indicador binario (1 si contiene URL, 0 si no)
- **Es retweet**: Indicador binario (1 si comienza con "RT @", 0 si no)
- **Longitud del texto procesado**: Número de caracteres después del preprocesamiento
- **Número de palabras procesadas**: Conteo de palabras después del preprocesamiento

### Características Temporales

A partir de la columna `tweet_created`:

- **Hora del día**: Hora en que se publicó el tweet (0-23)
- **Día, mes, año**: Componentes de la fecha
- **Día de la semana**: Número de día (0=lunes, 6=domingo)
- **Es fin de semana**: Indicador binario (1 si es sábado o domingo, 0 si no)

## Técnicas de Balanceo de Clases

Para corregir el desbalance entre clases de sentimientos, se implementaron cuatro estrategias:

1. **Submuestreo (Undersample)**: Reduce el tamaño de las clases mayoritarias seleccionando aleatoriamente observaciones hasta igualar la clase minoritaria.

2. **Sobremuestreo (Oversample)**: Aumenta el tamaño de las clases minoritarias duplicando observaciones aleatoriamente hasta igualar la clase mayoritaria.

3. **Híbrido**: Combina ambas técnicas, estableciendo un tamaño objetivo intermedio entre la clase minoritaria y la mayoritaria.

4. **Oversampling Puro**: A diferencia del sobremuestreo estándar, esta estrategia mantiene todas las muestras originales y solo aumenta las clases minoritarias mediante duplicación. No se eliminan datos de ninguna clase, lo que conserva toda la información original.

## Datasets Generados

### 1. Dataset Completo (`sentiment_processed_full.csv`)

- **Descripción**: Versión completa del dataset con todas las características adicionales
- **Características**: Mantiene la distribución original de clases
- **Contenido**: Incluye tanto el texto original como el procesado, más todas las características extraídas
- **Uso recomendado**: Análisis exploratorio detallado y experimentos que requieran preservar la distribución original

### 2. Dataset Balanceado (`sentiment_processed_balanced.csv`)

- **Descripción**: Dataset con clases de sentimiento balanceadas
- **Método de balanceo**: Híbrido
- **Distribución**:
  - Negativo: 5,364 tweets
  - Neutral: 5,364 tweets
  - Positivo: 5,364 tweets
- **Características incluidas**: Selección de las más relevantes
- **Uso recomendado**: Entrenamiento de modelos de clasificación multiclase

### 3. Dataset Binario (`sentiment_processed_binary.csv`)

- **Descripción**: Dataset que incluye solo tweets positivos y negativos
- **Método de balanceo**: Submuestreo
- **Distribución**:
  - Negativo: 2,236 tweets
  - Positivo: 2,236 tweets
- **Uso recomendado**: Entrenamiento de clasificadores binarios simples

### 4. Dataset con Oversampling Puro (`sentiment_processed_oversampled.csv`)

- **Descripción**: Dataset binario (positivo/negativo) balanceado mediante oversampling puro
- **Método de balanceo**: Oversampling Puro (sin reducción de datos)
- **Distribución**:
  - Negativo: 8,493 tweets (todos originales)
  - Positivo: 8,493 tweets (2,236 originales + 6,257 duplicados)
- **Características incluidas**: Todas las características del dataset completo
- **Uso recomendado**: Entrenamiento de clasificadores binarios que requieran preservar todos los ejemplos originales

## Visualización de Distribuciones

Se generó un gráfico comparativo (`sentiment_distributions.png`) que muestra la distribución de sentimientos en los cuatro datasets:
- Distribución original (desbalanceada)
- Distribución balanceada (método híbrido)
- Distribución binaria (solo positivo/negativo con submuestreo)
- Distribución con oversampling puro (mantiene todos los datos originales)

## Implementación

El preprocesamiento se implementó en Python utilizando las siguientes bibliotecas:
- `pandas`: Manipulación de datos tabulares
- `numpy`: Operaciones numéricas
- `nltk`: Procesamiento de lenguaje natural
- `re`: Expresiones regulares para limpieza de texto
- `sklearn.utils`: Funciones para el balanceo de clases
- `matplotlib`: Visualización de datos

## Consideraciones para Análisis Posteriores

1. **Elección del dataset**: Seleccionar el dataset apropiado según el objetivo del análisis:
   - Para exploración: Dataset completo
   - Para clasificación multiclase: Dataset balanceado
   - Para clasificación binaria con datos reducidos: Dataset binario
   - Para clasificación binaria preservando todos los datos: Dataset con oversampling puro

2. **Evaluación del modelo**: Al evaluar modelos entrenados con datos balanceados, es importante validarlos también con la distribución original para evaluar su rendimiento en condiciones reales.

3. **Características temporales**: Las características temporales extraídas pueden ser útiles para análisis de tendencias o para detectar patrones en la evolución de sentimientos.

4. **Candidatos y temas**: La relación entre candidatos, temas y sentimientos puede explorarse más a fondo utilizando las características extraídas.

El preprocesamiento realizado proporciona una base sólida para análisis posteriores, ofreciendo diferentes versiones del dataset adaptadas a distintas necesidades y corrigiendo los principales problemas identificados en el análisis exploratorio.
