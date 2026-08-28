# 10. Hoja de ruta: qué sigue para volverte experto

Este proyecto ya cubre una parte sólida del ML supervisado clásico. Esta hoja de ruta ordena lo que falta, de más inmediato a más avanzado, para que sigas "nutriendo" el proyecto de forma progresiva. Cada vez que implementes algo de esta lista, agrega su propio capítulo en esta carpeta siguiendo el estilo de los archivos `03`-`08`.

## 10.1 Corto plazo — completar lo que ya empezaste

Estos son literalmente huecos que ya existen en el código actual:

- [ ] **Implementar KNN de verdad** (`handle_knn_classification` está vacío). Ver esqueleto en [07-regresion-logistica-y-knn.md](07-regresion-logistica-y-knn.md).
- [ ] **Arreglar la ruta duplicada** de `/logistic-regression-classification` para que KNN sea alcanzable.
- [ ] **Corregir el `fit_transform` de test** en `handle_logistic_classification` (data leakage). Ver [07](07-regresion-logistica-y-knn.md).
- [ ] **Agregar escalado a SVR** (`svr_regression`). Ver [05](05-regresion-svr.md).
- [ ] **Arreglar rutas absolutas hardcodeadas** a los CSV — usar rutas relativas al proyecto. Ver [01](01-arquitectura-del-proyecto.md).

## 10.2 Clasificación — algoritmos que faltan

- **Naive Bayes** (`GaussianNB`, `MultinomialNB`): modelo probabilístico simple y rápido, muy usado como baseline en clasificación de texto (ej. detección de spam). Buen contraste con Regresión Logística/KNN porque asume independencia entre features.
- **SVM para clasificación** (`SVC`): ya conoces SVR (regresión); `SVC` es su contraparte de clasificación — mismo concepto de kernels aplicado a encontrar la frontera de máximo margen entre clases.
- **DecisionTreeClassifier / RandomForestClassifier**: versión de clasificación de los modelos que ya implementaste para regresión en `housing.csv`. Aplícalos sobre `Social_Network_Ads.csv` y compara con Regresión Logística.
- **Clasificación multiclase completa de MNIST** (los 10 dígitos, no solo "es 5 o no"). Ver [08](08-clasificacion-mnist-y-metricas.md#87-para-seguir-practicando).

## 10.3 Aprendizaje no supervisado — el siguiente gran bloque

Hoy el proyecto es 100% supervisado. `Mall_Customers.csv` ya está en `sample_data/` sin usar — es el dataset perfecto para arrancar aquí:

- **K-Means (clustering)**: agrupa datos en `k` grupos según similitud, sin necesitar una etiqueta `Y`. Ejercicio clásico: segmentar clientes de `Mall_Customers.csv` por edad/ingreso/puntaje de gasto en grupos con comportamiento similar (segmentación de clientes — un caso de negocio real).
- **Método del codo (elbow method)**: técnica para elegir el número óptimo de clusters `k` en K-Means.
- **PCA (Principal Component Analysis)**: reduce la cantidad de variables (dimensiones) conservando la mayor información posible — útil tanto para visualizar datos de alta dimensión en 2D, como para acelerar/mejorar otros modelos (ej. aplicarlo antes de KNN, que sufre con muchas dimensiones — el "curse of dimensionality").
- **Clustering jerárquico**: alternativa a K-Means que no requiere elegir `k` de antemano, produce un dendrograma.

## 10.4 Mejora de modelos — llevar lo que ya tienes a nivel profesional

- **Regularización (Ridge, Lasso, ElasticNet)**: variantes de regresión lineal que penalizan coeficientes grandes para reducir overfitting, especialmente útil cuando hay muchas features correlacionadas (como en `housing.csv`). Aplícalo sobre los ejercicios de [03](03-regresion-lineal-simple-y-multiple.md) y [06](06-arboles-de-decision-y-random-forest.md).
- **GridSearchCV / RandomizedSearchCV**: búsqueda automática del mejor hiperparámetro (ej. `max_depth` del árbol, `k` de KNN, `degree` del polinomio) en vez de probar manualmente. Reemplaza el loop manual que ya usas en `tree_regression` por una búsqueda sistemática.
- **`sklearn.pipeline.Pipeline`**: encadena preprocesamiento (escalado, encoding) + modelo en un solo objeto, evitando el error de `fit_transform` en test que ya identificaste en [07](07-regresion-logistica-y-knn.md) — con un `Pipeline`, ese error es estructuralmente imposible de cometer.
- **`ColumnTransformer`**: aplica transformaciones distintas a distintas columnas dentro de un mismo `Pipeline` (ej. `StandardScaler` a columnas numéricas + `OneHotEncoder` a categóricas, en un solo paso) — resuelve de forma elegante el patrón que hoy haces "a mano" en `housing.csv` y `Social_Network_Ads.csv`.
- **Feature importance / interpretabilidad**: `feature_importances_` en árboles/Random Forest, coeficientes en modelos lineales, y herramientas más avanzadas como **SHAP** para explicar predicciones individuales.

## 10.5 Persistencia y despliegue — cerrar el ciclo de un modelo real

Hoy cada endpoint entrena el modelo **desde cero en cada petición HTTP** — funciona para aprender, pero no es cómo se usa ML en producción:

- **Guardar y cargar modelos entrenados** (`joblib.dump` / `joblib.load`, o `pickle`): entrenar una vez (offline, en un script/notebook), guardar el modelo, y que la API solo lo cargue y prediga — mucho más rápido y es el patrón estándar en producción.
- **Versionado de modelos**: cuando reentrenas con datos nuevos, cómo llevar control de qué versión de modelo está sirviendo cada endpoint.
- **Endpoint de predicción vs. endpoint de entrenamiento**: separar `POST /train` (entrena y guarda el modelo) de `POST /predict` (carga el modelo guardado y predice) — hoy ambos pasos están mezclados en cada método.

## 10.6 Fundamentos matemáticos (para entender el "por qué", no solo el "cómo")

Si quieres ir más allá de usar scikit-learn como caja negra:

- **Álgebra lineal**: vectores, matrices, producto punto — la base de cómo se representan `X` e `Y` internamente.
- **Cálculo/gradientes**: cómo funciona el descenso de gradiente que usa `SGDClassifier` (§8.3) y, en el fondo, casi todo entrenamiento de modelos.
- **Probabilidad y estadística**: distribución normal, varianza, correlación (ya la usaste en `housing.csv`) — base de Naive Bayes y de la interpretación de métricas.
- **Implementar un algoritmo desde cero**: el ejercicio más formativo de todos. Por ejemplo, programar regresión lineal simple con descenso de gradiente manual (sin scikit-learn) usando solo NumPy, y comparar el resultado contra `LinearRegression()` de scikit-learn — deberían coincidir, y entenderás exactamente qué hace `.fit()` por dentro.

## 10.7 Sugerencia de orden concreto

Si quieres una secuencia recomendada para las próximas semanas, este es un orden razonable:

1. Arreglar los huecos de §10.1 (son rápidos y consolidan lo ya aprendido).
2. Implementar K-Means sobre `Mall_Customers.csv` — primer contacto con no supervisado.
3. Aprender `Pipeline` + `ColumnTransformer` y refactorizar `handle_logistic_classification` con ellos — consolida el concepto de data leakage de forma práctica.
4. Agregar Naive Bayes y SVC sobre `Social_Network_Ads.csv`, comparar los 4-5 modelos de clasificación entre sí con una tabla de métricas.
5. Agregar `GridSearchCV` a Random Forest sobre `housing.csv`.
6. Implementar guardado/carga de modelos con `joblib` para al menos un endpoint.

Cada vez que completes un punto, dime y actualizamos esta hoja de ruta y agregamos el capítulo correspondiente en esta carpeta.
