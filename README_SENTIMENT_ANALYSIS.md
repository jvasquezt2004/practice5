# Análisis de Sentimiento: Mejoras y Comparación

Este documento detalla las mejoras realizadas al script original de análisis de sentimiento (`sentiment(1).py`), las diferentes versiones implementadas, y la comparación de resultados entre los distintos enfoques.

## Versiones de Scripts Implementados

### 1. Script Original (`sentiment(1).py`)
- **Descripción**: Implementación original que utiliza NLTK con un clasificador Naive Bayes simple.
- **Características**:
  - Preprocesamiento básico (minúsculas, eliminación de URLs, menciones, hashtags)
  - Extracción de características basada en presencia de palabras
  - Sin métricas de evaluación detalladas
  - Uso del dataset original con desbalance de clases

### 2. Script Original con Métricas (`sentiment_original_with_metrics.py`)
- **Descripción**: Versión modificada del script original que añade métricas de evaluación.
- **Características**:
  - Mantiene el mismo algoritmo y preprocesamiento que el original
  - Añade cálculo de precisión por clase y general
  - Incluye visualización de resultados con gráficos
  - Permite la comparación directa con las versiones mejoradas

### 3. Script Mejorado Ligero (`sentiment_improved_light.py`)
- **Descripción**: Versión optimizada para reducir el uso de recursos y evitar fallos de memoria.
- **Características**:
  - Muestreo estratificado (hasta 1000 tweets por clase)
  - Preprocesamiento mejorado (lemmatización, filtrado de stopwords)
  - Conjunto de características limitado a las 500 palabras más frecuentes
  - Uso de clasificadores Naive Bayes y MultinomialNB
  - Generación de nubes de palabras y métricas detalladas

### 4. Script Mejorado Simple (`sentiment_improved_simple.py`)
- **Descripción**: Versión final optimizada que utiliza oversampling puro y simplifica el modelo.
- **Características**:
  - Utiliza el dataset con oversampling puro (`sentiment_processed_oversampled.csv`)
  - Elimina el clasificador MultinomialNB, manteniendo solo Naive Bayes
  - Añade características adicionales (longitud del texto, presencia de palabras positivas/negativas)
  - Preprocesamiento completo (URLs, menciones, puntuación, números, stopwords, lematización)
  - Conjunto de características ampliado a las 2000 palabras más frecuentes
  - Evaluación detallada y visualización de resultados

## Comparación de Resultados

### Precisión General

| Script | Precisión General |
|--------|-------------------|
| Original | 83.22% |
| Mejorado con Oversampling | 78.34% |

### Precisión por Clase

| Script | Precisión Negativos | Precisión Positivos |
|--------|---------------------|---------------------|
| Original | 94.29% | 41.16% |
| Mejorado con Oversampling | 74.75% | 81.93% |

### Distribución de Datos de Prueba

| Script | Tweets Negativos | Tweets Positivos |
|--------|------------------|------------------|
| Original | 1,699 | 447 |
| Mejorado con Oversampling | 1,699 | 1,699 |

## Análisis de las Diferencias

### 1. Sesgo hacia la Clase Mayoritaria

El script original muestra una clara tendencia a clasificar tweets como negativos (la clase mayoritaria), lo que resulta en:
- Alta precisión para tweets negativos (94.29%)
- Baja precisión para tweets positivos (41.16%)
- Precisión general artificialmente alta debido al desbalance de clases

### 2. Equilibrio vs. Precisión General

El script mejorado con oversampling puro:
- Reduce la precisión general al 78.34% (-4.88%)
- Mejora dramáticamente la precisión de tweets positivos al 81.93% (+40.77%)
- Ofrece un rendimiento más equilibrado entre ambas clases
- Reduce el sesgo hacia la clase mayoritaria

### 3. Ejemplos de Clasificación

#### Texto: "This was the best political event I've seen in years"
- **Original**: Clasificado como Negativo (98.21% de confianza)
- **Mejorado**: Clasificado como Positivo (99.57% de confianza)

#### Texto: "Complete disaster, waste of time watching these people"
- **Original**: Clasificado como Negativo (99.88% de confianza)
- **Mejorado**: Clasificado como Positivo (57.50% de confianza) - Clasificación incorrecta

## Factores que Contribuyeron a las Mejoras

### 1. Balanceo de Datos
- El oversampling puro preserva toda la información original mientras crea un conjunto de entrenamiento balanceado
- No se pierde información valiosa de la clase mayoritaria
- El clasificador aprende patrones más equilibrados

### 2. Preprocesamiento Mejorado
- Limpieza más exhaustiva del texto
- Eliminación efectiva de elementos irrelevantes
- Lematización para reducir variantes de palabras

### 3. Características Adicionales
- Incorporación de características como longitud del texto
- Inclusión de indicadores de presencia de palabras positivas/negativas comunes
- Ampliación del vocabulario de características a 2000 palabras

### 4. Evaluación Más Robusta
- Medición separada de la precisión por clase
- Visualización de resultados para mejor interpretación
- Ejemplos de clasificación con niveles de confianza

## Conclusiones

1. **¿Cuál es mejor?** Depende del objetivo:
   - Si se valora la precisión general y detectar principalmente tweets negativos: el original
   - Si se busca un clasificador equilibrado que funcione bien en ambas clases: el mejorado

2. **Ventajas del enfoque mejorado**:
   - Mayor equilibrio entre clases
   - Mejor precisión en la clase minoritaria (positivos)
   - Menor tendencia a clasificar todo como negativo
   - Mejor rendimiento en ejemplos reales con lenguaje positivo

3. **Ventajas del enfoque original**:
   - Mayor precisión general
   - Excelente detección de tweets negativos
   - Procesamiento más simple

4. **Recomendaciones**:
   - Para una aplicación real, el enfoque mejorado probablemente sea más útil a pesar de su menor precisión general
   - El uso de oversampling puro es una estrategia efectiva para manejar el desbalance de clases sin perder información
   - La simplificación del modelo (eliminando MultinomialNB) mantiene un buen rendimiento mientras reduce la complejidad

Este análisis demuestra cómo las técnicas de balanceo de clases y las mejoras en el preprocesamiento pueden tener un impacto significativo en el rendimiento de un clasificador de sentimientos, especialmente cuando se evalúa su capacidad para manejar clases minoritarias de manera efectiva.
