# 6. Árboles de decisión, Random Forest y el pipeline de `housing.csv`

> Requiere: [02](02-fundamentos-de-machine-learning.md), especialmente escalado/codificación.

Este capítulo cubre **tres endpoints** (`housing_linear_regression`, `decision_tree_regression`, `random_forest_regression`) porque los tres comparten el mismo dataset y el mismo pipeline de preparación de datos — solo cambia el modelo final. Es, con diferencia, el ejercicio más completo del proyecto en cuanto a *feature engineering*.

## 6.1 Dataset usado: `housing.csv`

El dataset de precios de viviendas en California (usado en el libro *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, de Aurélien Géron — vale la pena saber que este pipeline está inspirado directamente en el capítulo 2 de ese libro). Columnas relevantes: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value` (el target) y `ocean_proximity` (categórica).

## 6.2 El pipeline compartido, paso a paso

📍 `machine_learning/services/shared/housing_preprocessing.py` — usado por los tres servicios de `services/regression/tree_ensemble_service.py`. Antes este bloque estaba copiado casi idéntico 3 veces; ahora vive en un solo lugar (ver [01](01-arquitectura-del-proyecto.md#15-hallazgos-corregidos-en-esta-refactorización)).

### Paso 1 — Ingeniería de atributos (*feature engineering*)

```python
data_for_corr['rooms_per_household'] = data_for_corr['total_rooms'] / data_for_corr['households']
data_for_corr['bedrooms_per_room'] = data_for_corr['total_bedrooms'] / data_for_corr['households']
data_for_corr['population_per_household'] = data_for_corr['population'] / data_for_corr['households']
```

Esto es **feature engineering**: crear nuevas columnas combinando las existentes para que capturen mejor la información relevante. `total_rooms` por sí solo no dice mucho (una zona con más casas tendrá más cuartos totales, sin que eso signifique casas más grandes) — pero `rooms_per_household` (cuartos por hogar) sí es una medida útil del tamaño típico de vivienda en esa zona. Es una de las técnicas más importantes en ML práctico: a menudo mejora más el modelo crear una buena feature que cambiar de algoritmo.

### Paso 2 — Manejo de valores faltantes

```python
data_for_corr["total_bedrooms"] = data_for_corr["total_bedrooms"].fillna(
    data_for_corr["total_bedrooms"].median()
)
```

`total_bedrooms` tiene valores nulos en el dataset original. La estrategia usada es **imputación por mediana**: rellenar los huecos con el valor mediano de esa misma columna. Se prefiere la mediana sobre la media porque es más robusta a valores extremos (outliers). Alternativas que existen (no usadas aquí): eliminar las filas con nulos (`dropna`), eliminar la columna completa, o imputar con un modelo predictivo.

### Paso 3 — Codificación de la variable categórica

```python
encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[["ocean_proximity"]])
encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out())
```

`ocean_proximity` toma valores como `NEAR BAY`, `INLAND`, `<1H OCEAN`, etc. — categorías sin orden natural, por eso `OneHotEncoder` (no `OrdinalEncoder`) es la elección correcta, tal como se explicó en [02](02-fundamentos-de-machine-learning.md). El resultado es un `DataFrame` con una columna binaria por categoría (ej. `ocean_proximity_INLAND`, `ocean_proximity_NEAR BAY`...).

> La versión original del código también ejecutaba un `OrdinalEncoder` sobre la misma columna solo con fines exploratorios (nunca alimentaba ningún modelo). Se quitó del pipeline compartido para mantenerlo enfocado — si quieres ver la comparación `OrdinalEncoder` vs `OneHotEncoder` con tus propios ojos, es un buen mini-ejercicio: agrégalo temporalmente en un notebook aparte con `data[['ocean_proximity']]`.

### Paso 4 — Análisis de correlación (visible en `housing_linear_regression`)

```python
corr_matrix = data_for_corr.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
```

La matriz de correlación mide, para cada par de columnas numéricas, qué tan relacionadas están linealmente (valores de -1 a 1). Ordenar por correlación contra `median_house_value` te dice **qué variables tienen más relación lineal con el precio** — es una forma rápida de priorizar qué features probar primero en el modelo (por eso, en el paso 5, `median_income` es la primera columna que se agrega: es la que más correlaciona con el precio).

> Importante: correlación mide relación **lineal**. Una variable puede ser muy predictiva de forma no lineal y tener correlación baja — por eso este análisis es un punto de partida, no la palabra final.

### Paso 5 — Selección incremental de features

📍 `_incremental_column_scores()` en `tree_ensemble_service.py` — factoriza el loop que antes estaba repetido en los tres métodos.

```python
def _incremental_column_scores(model_factory, data_for_corr, encoded_df):
    y = data_for_corr["median_house_value"].values
    columns_used = []
    scores = []

    for col in HOUSING_MODEL_COLUMNS:
        columns_used.append(col)
        X = pd.concat([data_for_corr[columns_used], encoded_df], axis=1).values
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = model_factory()          # LinearRegression() / DecisionTreeRegressor(...) / RandomForestRegressor(...)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        scores.append({"columns": list(columns_used), "r2_score": r2_score(Y_test, y_pred)})

    return scores
```

En vez de entrenar un solo modelo con todas las columnas de una vez, **entrena un modelo nuevo en cada iteración, agregando una columna más cada vez** (siempre concatenando las columnas de `ocean_proximity` codificadas). La respuesta del endpoint incluye `scores_by_columns`, una lista con el `R²` obtenido en cada paso — puedes ver directamente en el JSON cómo cambia el `R²` a medida que agregas más información: un experimento de **selección de features**.

Vas a notar, por ejemplo, que `median_income` sola ya explica una buena parte del precio (es la variable más predictiva, consistente con el análisis de correlación del paso 4), y que agregar `latitude`/`longitude` casi siempre ayuda bastante más (porque en California, la ubicación geográfica está muy ligada al precio de la vivienda — cercanía a la costa, a ciudades como San Francisco, etc.).

> `random_state=42` ahora está fijo en este `train_test_split` (antes no lo estaba, así que cada corrida daba un `R²` ligeramente distinto) — puedes comparar resultados entre corridas de forma justa.

## 6.3 Los tres modelos que se entrenan sobre este mismo pipeline

### `housing_linear_regression` (`POST /housing-linear-regression`) → Regresión Lineal
Usa `LinearRegression()` como línea base (baseline): antes de probar modelos más complejos, es buena práctica saber qué tan bien funciona el modelo más simple posible. También genera los tres gráficos exploratorios (§6.4).

### `decision_tree_regression` (`POST /decision-tree-regression`) → Árbol de Decisión

**Teoría**: un árbol de decisión divide el espacio de datos en regiones haciendo preguntas tipo "¿`median_income` > 5.2?" de forma recursiva, hasta llegar a hojas donde predice el promedio de `Y` de los ejemplos que cayeron ahí. A diferencia de la regresión lineal, puede capturar relaciones **no lineales** y no necesita escalado de variables (no le importa que `total_rooms` esté en miles y `latitude` en decenas — solo compara valores dentro de la misma columna).

**Riesgo**: un árbol sin restricciones (sin `max_depth`) tiende a **overfitting** severo — puede crecer hasta tener una hoja por cada ejemplo de entrenamiento, memorizando el dataset. El endpoint ahora acepta `max_depth` en el body (`TreeRegressionSchema`, default `None` = sin límite):

```bash
curl -X POST http://localhost:8080/decision-tree-regression -H "Content-Type: application/json" -d '{"max_depth": 8}'
```

Compara el `r2_score` del último paso (`scores_by_columns[-1]`) con `max_depth=None` vs `max_depth=8` vs `max_depth=3` — vas a ver el trade-off underfitting/overfitting directamente en los números.

### `random_forest_regression` (`POST /random-forest-regression`) → Random Forest

**Teoría**: en vez de un solo árbol, entrena **muchos árboles** (`n_estimators`, default 100), cada uno sobre una muestra aleatoria distinta de los datos (y de las features en cada división) — técnica llamada ***bagging*** (Bootstrap Aggregating). La predicción final es el **promedio** de las predicciones de todos los árboles.

**Por qué funciona mejor que un árbol solo**: cada árbol individual puede sobre-ajustarse a su muestra particular de datos, pero como cada uno se equivoca de forma distinta (aleatoria), al promediar sus errores tienden a cancelarse — el conjunto (*ensemble*) generaliza mejor que cualquier árbol individual. Es la idea de "la sabiduría de las masas" aplicada a modelos.

El endpoint acepta `n_estimators` y `max_depth` (`RandomForestRegressionSchema`):

```bash
curl -X POST http://localhost:8080/random-forest-regression -H "Content-Type: application/json" -d '{"n_estimators": 50, "max_depth": 10}'
```

**Comparación verificada en este proyecto** (usando todas las columnas, `n_estimators=50`): `R²` Random Forest ≈ **0.81**, Árbol de decisión (`max_depth=6`) ≈ **0.66**, Regresión lineal ≈ **0.60** — el orden esperado se cumple: `RandomForest > DecisionTree > LinearRegression`, porque Random Forest combina la capacidad de capturar no-linealidad (como el árbol) con menor varianza (por el promediado).

## 6.4 Gráficos generados (solo en `housing_linear_regression`)

La respuesta trae `plot_files` con el nombre real de cada archivo generado (uno único por request, dentro de `results_graphics/`):

- `plot_files.scatter_plot` (ej. `scatter_plot_20260901_a1b2c3d4.png`): mapa de California donde cada punto es una vivienda, coloreado por `median_house_value` y con tamaño proporcional a `population` — permite *ver* geográficamente dónde están las zonas caras.
- `plot_files.correlation_plot`: heatmap de la matriz de correlación (paso 4, pero visual).
- `plot_files.histograms`: distribución de cada variable numérica — útil para detectar asimetrías, outliers o columnas con "topes" artificiales (ej. `median_house_value` suele tener un pico en el valor máximo por censura de datos en el dataset original).

## 6.5 Para seguir practicando

- Imprime `forest_reg.feature_importances_` después de entrenar el Random Forest (puedes agregarlo temporalmente en `random_forest_regression`) — te dice qué columnas usó más el modelo para decidir, y es una forma directa de "explicar" un modelo de caja relativamente negra.
- Compara sistemáticamente varias combinaciones de `n_estimators`/`max_depth` llamando al endpoint varias veces — luego automatiza esa búsqueda con `GridSearchCV` (ver [10-hoja-de-ruta.md](10-hoja-de-ruta.md)).
- Con el pipeline ya factorizado en `services/shared/housing_preprocessing.py`, es un buen ejercicio agregar un cuarto modelo (ej. `GradientBoostingRegressor`) reutilizando `prepare_housing_dataset()` y `_incremental_column_scores()` — deberías poder hacerlo en pocas líneas.
