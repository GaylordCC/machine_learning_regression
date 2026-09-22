# Documentación de estudio — Machine Learning

Esta carpeta es el **material de estudio** del proyecto. Cada técnica implementada tiene aquí su explicación teórica, su conexión con el código y ejercicios para profundizar.

No es documentación de "cómo usar la API" (para eso está el `README.md` de la raíz) — es documentación para **entender la teoría detrás de cada parte del código**.

## Cómo usar esta carpeta

1. Sigue el orden numerado la primera vez que estudies. Cada archivo asume conceptos de los anteriores.
2. Cuando se implemente una técnica nueva en el proyecto, se agrega un archivo nuevo aquí siguiendo la misma estructura: **Teoría → Código en este proyecto → Buenas prácticas / errores comunes → Para seguir practicando**.
3. Los archivos citan las rutas de los archivos y funciones del código para ir y venir entre la teoría y la implementación real.

## Índice de contenidos

| # | Archivo | Tema |
|---|---------|------|
| 1 | [01-arquitectura-del-proyecto.md](01-arquitectura-del-proyecto.md) | Cómo está organizado el proyecto (routers, services, models) y decisiones de diseño |
| 2 | [02-fundamentos-de-machine-learning.md](02-fundamentos-de-machine-learning.md) | Conceptos base: tipos de aprendizaje, train/test, overfitting, métricas |
| 3 | [03-regresion-lineal-simple-y-multiple.md](03-regresion-lineal-simple-y-multiple.md) | Regresión lineal simple y múltiple (`Advertising.csv`) |
| 4 | [04-regresion-polinomica.md](04-regresion-polinomica.md) | Regresión polinómica (dataset de salarios por posición) |
| 5 | [05-regresion-svr.md](05-regresion-svr.md) | Support Vector Regression (SVR) |
| 6 | [06-arboles-de-decision-y-random-forest.md](06-arboles-de-decision-y-random-forest.md) | Árboles de decisión, Random Forest y el pipeline de `housing.csv` |
| 7 | [07-regresion-logistica-y-knn.md](07-regresion-logistica-y-knn.md) | Clasificación: Regresión Logística y K-Nearest Neighbors |
| 8 | [08-clasificacion-mnist-y-metricas.md](08-clasificacion-mnist-y-metricas.md) | Clasificación de imágenes (MNIST), validación cruzada y métricas |
| 9 | [09-glosario.md](09-glosario.md) | Glosario de términos de ML en español |
| 10 | [10-hoja-de-ruta.md](10-hoja-de-ruta.md) | Qué falta por aprender/implementar para seguir creciendo el proyecto |
| 11 | [11-pipeline-de-machine-learning.md](11-pipeline-de-machine-learning.md) | El pipeline completo: extracción, limpieza y preparación, división, entrenamiento, evaluación y entrega |
| 12 | [12-metricas-de-evaluacion.md](12-metricas-de-evaluacion.md) | Métricas de evaluación (R², MSE, RMSE, MAE, matriz de confusión, accuracy, precision, recall, F1, ROC-AUC): cómo se usan, comparación, casos de uso, flujos y el proceso de evaluación del proyecto |

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
