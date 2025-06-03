# Análisis del Dataset Sentiment(1)

Este documento resume el análisis realizado sobre el dataset de sentimientos de tweets relacionados con el debate presidencial del Partido Republicano (GOP) en 2015.

## Información Básica del Dataset

- **Nombre del archivo**: `Sentiment(1) 1.csv`
- **Número total de tweets**: 13,871
- **Columnas principales**: id, candidate, sentiment, subject_matter, text, tweet_created, etc.
- **Período de tiempo**: Principalmente 6-7 de agosto de 2015

## Distribución de Sentimientos

El análisis muestra que el dataset **NO está balanceado** en términos de sentimientos:

| Sentimiento | Cantidad | Porcentaje |
|-------------|----------|------------|
| Negativo    | 8,493    | 61.2%      |
| Neutral     | 3,142    | 22.7%      |
| Positivo    | 2,236    | 16.1%      |

Esta desbalance significativo hacia los sentimientos negativos puede afectar el rendimiento de modelos de clasificación.

## Distribución de Candidatos Mencionados

Los candidatos más mencionados en el dataset son:

| Candidato                 | Cantidad | Porcentaje |
|---------------------------|----------|------------|
| No candidate mentioned    | 7,491    | 54.4%      |
| Donald Trump              | 2,813    | 20.4%      |
| Jeb Bush                  | 705      | 5.1%       |
| Ted Cruz                  | 637      | 4.6%       |
| Ben Carson                | 404      | 2.9%       |
| Mike Huckabee             | 393      | 2.8%       |
| Otros candidatos          | < 300 cada uno | < 2.2% cada uno |

Se observa un claro dominio de tweets que no mencionan candidatos específicos, seguido por Donald Trump que supera ampliamente al resto de candidatos.

## Distribución de Temas

Los temas discutidos en los tweets:

| Tema                               | Cantidad | Porcentaje |
|------------------------------------|----------|------------|
| None of the above                  | 8,148    | 59.8%      |
| FOX News or Moderators             | 2,900    | 21.3%      |
| Religion                           | 407      | 3.0%       |
| Foreign Policy                     | 366      | 2.7%       |
| Women's Issues (not abortion)      | 362      | 2.7%       |
| Racial issues                      | 353      | 2.6%       |
| Otros temas                        | < 300 cada uno | < 2.2% cada uno |

La mayoría de los tweets no se asocian a temas específicos, seguidos por aquellos relacionados con Fox News y los moderadores del debate.

## Relación entre Candidatos y Sentimientos

| Candidato | Negativo | Neutral | Positivo | % Positivo |
|-----------|----------|---------|----------|------------|
| Donald Trump | 1,758 | 446 | 609 | 21.7% |
| Ted Cruz | 221 | 126 | 290 | 45.5% |
| John Kasich | 82 | 47 | 113 | 46.7% |
| Marco Rubio | 105 | 51 | 119 | 43.3% |
| Ben Carson | 186 | 54 | 164 | 40.6% |
| No candidate mentioned | 4,724 | 2,087 | 680 | 9.1% |
| Jeb Bush | 589 | 72 | 44 | 6.2% |

Ted Cruz, John Kasich y Marco Rubio tienen las proporciones más altas de sentimientos positivos, mientras que Jeb Bush tiene la proporción más baja.

## Relación entre Temas y Sentimientos

| Tema | Negativo | Neutral | Positivo | % Positivo |
|------|----------|---------|----------|------------|
| None of the above | 4,226 | 2,313 | 1,609 | 19.7% |
| FOX News or Moderators | 2,230 | 316 | 354 | 12.2% |
| Abortion | 220 | 45 | 28 | 9.6% |
| Women's Issues (not abortion) | 324 | 28 | 10 | 2.8% |
| Religion | 283 | 104 | 20 | 4.9% |
| Racial issues | 292 | 41 | 20 | 5.7% |

Los temas relacionados con asuntos de mujeres, religión y raza tienen las proporciones más bajas de sentimientos positivos.

## Análisis Temporal

| Fecha | Cantidad de Tweets |
|-------|-------------------|
| 2015-08-06 | 5,303 |
| 2015-08-07 | 8,568 |

La mayoría de los tweets fueron publicados el 7 de agosto, el día después del debate.

## Valores Faltantes

| Columna | Valores Nulos |
|---------|---------------|
| candidate_gold | 13,843 |
| relevant_yn_gold | 13,839 |
| sentiment_gold | 13,856 |
| subject_matter_gold | 13,853 |
| tweet_coord | 13,850 |
| tweet_location | 3,912 |
| user_timezone | 4,403 |

Las columnas "gold" (etiquetas de referencia) están casi completamente vacías, lo que sugiere que fueron incluidas para futuros trabajos de anotación.

## Conclusiones del Análisis

1. **Desbalance significativo**: El dataset muestra un fuerte sesgo hacia sentimientos negativos (61.2%), lo que puede afectar el entrenamiento de modelos de clasificación.

2. **Dominancia de Donald Trump**: Es el candidato más mencionado por amplio margen, reflejando su protagonismo mediático durante el debate.

3. **Foco en los moderadores**: Después de tweets sin tema específico, los relacionados con Fox News y los moderadores son los más frecuentes.

4. **Sentimientos por candidato**: Ted Cruz, John Kasich y Marco Rubio recibieron proporciones más altas de sentimientos positivos.

5. **Temas controversiales**: Temas como asuntos de mujeres, religión y raza tienden a generar sentimientos más negativos.

6. **Temporalidad**: El volumen de tweets aumentó significativamente el día después del debate.

7. **Datos faltantes**: Hay una cantidad significativa de valores nulos en ciertas columnas, especialmente en las etiquetas "gold".

Estos hallazgos son fundamentales para entender la naturaleza del dataset y guiar estrategias de preprocesamiento adecuadas antes de desarrollar modelos de análisis de sentimientos.
