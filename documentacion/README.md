# Documentación de estudio — Machine Learning

Esta carpeta es tu **material de estudio personal**, construido a partir del código real de este proyecto. La idea es que cada técnica que ya implementaste (o que implementes en el futuro) tenga aquí su explicación teórica, su conexión con el código y ejercicios para profundizar.

No es documentación de "cómo usar la API" (para eso está el `README.md` de la raíz) — es documentación para **aprender la teoría detrás de cada línea de código que ya escribiste**.

## Cómo usar esta carpeta

1. Sigue el orden numerado la primera vez que estudies. Cada archivo asume conceptos de los anteriores.
2. Cuando implementes una técnica nueva en el proyecto, agrega un archivo nuevo aquí (o pide ayuda para generarlo) siguiendo la misma estructura: **Teoría → Código en este proyecto → Buenas prácticas / errores comunes → Para seguir practicando**.
3. Los archivos citan rutas y líneas exactas del código (`archivo.py:línea`) para que puedas ir y venir entre la teoría y la implementación real.

## Índice de contenidos

| # | Archivo | Tema |
|---|---------|------|
| 1 | [01-arquitectura-del-proyecto.md](01-arquitectura-del-proyecto.md) | Cómo está organizado el proyecto (routers, services, models) y hallazgos/mejoras |
| 2 | [02-fundamentos-de-machine-learning.md](02-fundamentos-de-machine-learning.md) | Conceptos base: tipos de aprendizaje, train/test, overfitting, métricas |
| 3 | [03-regresion-lineal-simple-y-multiple.md](03-regresion-lineal-simple-y-multiple.md) | Regresión lineal simple y múltiple (`Advertising.csv`) |
| 4 | [04-regresion-polinomica.md](04-regresion-polinomica.md) | Regresión polinómica (dataset de salarios por posición) |
| 5 | [05-regresion-svr.md](05-regresion-svr.md) | Support Vector Regression (SVR) |
| 6 | [06-arboles-de-decision-y-random-forest.md](06-arboles-de-decision-y-random-forest.md) | Árboles de decisión, Random Forest y el pipeline de `housing.csv` |
| 7 | [07-regresion-logistica-y-knn.md](07-regresion-logistica-y-knn.md) | Clasificación: Regresión Logística y K-Nearest Neighbors (pendiente) |
| 8 | [08-clasificacion-mnist-y-metricas.md](08-clasificacion-mnist-y-metricas.md) | Clasificación de imágenes (MNIST), validación cruzada y métricas |
| 9 | [09-glosario.md](09-glosario.md) | Glosario de términos de ML en español |
| 10 | [10-hoja-de-ruta.md](10-hoja-de-ruta.md) | Qué falta por aprender/implementar para seguir creciendo el proyecto |

## Mapa rápido: técnica ↔ archivo de código

| Técnica | Servicio / método | Endpoint |
|---|---|---|
| Análisis exploratorio (EDA) | `regression/linear_regression_service.py :: handle_user_query` | `POST /v1/machine-learning` |
| Regresión lineal simple | `regression/linear_regression_service.py :: regression_linear_model` | `POST /v1/linear-regression` |
| Regresión lineal múltiple | `regression/linear_regression_service.py :: regression_multi_linear_model` | `POST /v1/multi-linear-regression` |
| Regresión polinómica | `regression/polynomial_regression_service.py :: polynomical_regression` | `POST /v1/polynomial-regression` |
| SVR (Support Vector Regression) | `regression/svr_service.py :: svr_regression` | `POST /v1/svr-regression` |
| Regresión lineal iterativa sobre `housing.csv` | `regression/tree_ensemble_service.py :: housing_linear_regression` | `POST /v1/housing-linear-regression` |
| Árbol de decisión (regresión) | `regression/tree_ensemble_service.py :: decision_tree_regression` | `POST /v1/decision-tree-regression` |
| Random Forest (regresión) | `regression/tree_ensemble_service.py :: random_forest_regression` | `POST /v1/random-forest-regression` |
| Clasificación binaria de dígitos (MNIST) | `classification/image_classification_service.py :: handle_classification_image` | `POST /v1/classification-algorithm` |
| Regresión logística | `classification/logistic_regression_service.py :: handle_logistic_classification` | `POST /v1/logistic-regression-classification` |
| KNN | `classification/knn_service.py :: handle_knn_classification` | `POST /v1/knn-classification` |

Todos los archivos de `services/` viven bajo `machine_learning/services/`. El pipeline compartido de `housing.csv` y de `Social_Network_Ads.csv` está en `services/shared/`.

> Nota: en `03-08` vas a ver referencias a *"hallazgos"* — algunas ya están corregidas en el código (se documentó qué cambió en [01](01-arquitectura-del-proyecto.md#15-hallazgos-corregidos-en-esta-refactorización)); se dejaron explicadas igual porque son errores muy comunes y vale la pena que entiendas **por qué** eran un problema.
