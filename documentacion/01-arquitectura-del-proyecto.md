# 1. Arquitectura del proyecto

> Actualizado tras la refactorización de organización del proyecto: los servicios pasaron de 2 archivos gigantes a un paquete por técnica, y se corrigieron los bugs que se habían documentado aquí originalmente. Al final de este archivo (§1.5) queda el registro de qué se corrigió y por qué, como referencia de aprendizaje.

## 1.1 Visión general

El proyecto es una API en **FastAPI** que envuelve modelos de Machine Learning (hechos mientras estudiabas) como endpoints HTTP. Arquitectura en 3 capas:

```
Cliente HTTP (Swagger /docs, curl, frontend...)
        │
        ▼
┌───────────────────┐
│  Router             │  machine_learning/routers/*.py
│  (capa HTTP)        │  Define la URL, el método HTTP y el schema de entrada/salida
└─────────┬──────────┘
          │ instancia y llama
          ▼
┌───────────────────┐
│  Service            │  machine_learning/services/**/*.py
│  (capa de negocio)  │  Carga datos, entrena el modelo, calcula métricas, grafica
└─────────┬──────────┘
          │ lee / escribe
          ▼
┌───────────────────┐        ┌───────────────────┐
│  sample_data/*.csv │        │  results_graphics/  │
│  (datasets locales) │        │  (gráficos .png)    │
└───────────────────┘        └───────────────────┘
```

Además existe una cuarta capa, sin relación con ML: **persistencia relacional** (`models.py` + `database.py` + Alembic + PostgreSQL) para un modelo `User` — tu práctica de SQLAlchemy/Alembic, independiente de los servicios de ML.

## 1.2 Inventario de componentes (estructura actual)

```
machine_learning/
├── main.py                     # Crea la app FastAPI, registra routers y matplotlib.use("Agg")
├── core/
│   └── paths.py                # Rutas absolutas basadas en __file__ (sample_data, results_graphics)
├── schemas.py                  # Todos los contratos Pydantic de entrada (uno por endpoint configurable)
├── database.py / models.py     # SQLAlchemy (no relacionado con ML)
├── routers/
│   ├── regression.py           # 8 endpoints de regresión — capa delgada
│   └── classification.py       # 3 endpoints de clasificación — capa delgada
├── services/
│   ├── shared/
│   │   ├── housing_preprocessing.py       # Pipeline compartido de housing.csv
│   │   └── social_ads_preprocessing.py    # Pipeline compartido de Social_Network_Ads.csv
│   ├── regression/
│   │   ├── linear_regression_service.py     # EDA + regresión simple + múltiple
│   │   ├── polynomial_regression_service.py
│   │   ├── svr_service.py
│   │   └── tree_ensemble_service.py         # Lineal/Árbol/Random Forest sobre housing.csv
│   └── classification/
│       ├── logistic_regression_service.py
│       ├── knn_service.py
│       └── image_classification_service.py  # MNIST
└── sample_data/*.csv
```

Un archivo = una técnica. Esto reemplaza los dos archivos originales (`main_ml_service.py` con ~530 líneas y `classification_ml_service.py`) que mezclaban todo. Es más fácil de estudiar: cuando quieras repasar SVR, abres exactamente `services/regression/svr_service.py` y nada más.

### `machine_learning/core/paths.py`
Centraliza las rutas del proyecto usando `pathlib` y `Path(__file__).resolve()` — funciona sin importar desde qué directorio se lance `uvicorn`, y es idéntico corriendo local o en Docker. Reemplaza las rutas absolutas hardcodeadas que existían antes (ver §1.5).

### `machine_learning/services/shared/`
Cada dataset que se reutiliza en más de una técnica (`housing.csv`, `Social_Network_Ads.csv`) tiene aquí su función de preparación de datos compartida, para no repetir el mismo bloque de limpieza/encoding en cada service que lo use.

## 1.3 Flujo de una petición típica

Ejemplo con `POST /linear-regression`:

1. `routers/regression.py` recibe el POST, valida el body contra `RegressionSchema` (Pydantic).
2. Instancia `LinearRegressionService()` y llama a `regression_linear_model(request=request)`.
3. `services/regression/linear_regression_service.py`:
   - Carga `Advertising.csv` vía `core/paths.py` (ruta relativa al proyecto, no a tu máquina).
   - Arma `X` (la columna elegida) e `Y` (Sales).
   - Divide en train/test, entrena `LinearRegression`, calcula `RMSE`/`R²`.
   - Genera y guarda un gráfico en `results_graphics/`.
   - Devuelve `{"predictions": [...], "rmse": ..., "r2_score": ...}`.
4. FastAPI serializa la respuesta a JSON.

## 1.4 Endpoints disponibles hoy

| Método/Ruta | Service | Hiperparámetros configurables (body) |
|---|---|---|
| `POST /machine-learning` | `LinearRegressionService.handle_user_query` | — |
| `POST /linear-regression` | `LinearRegressionService.regression_linear_model` | `column_name`: TV\|Radio\|Newspaper |
| `POST /multi-linear-regression` | `LinearRegressionService.regression_multi_linear_model` | — |
| `POST /polynomial-regression` | `PolynomialRegressionService.polynomical_regression` | `degree` (1-10, default 4) |
| `POST /svr-regression` | `SvrRegressionService.svr_regression` | `kernel`: linear\|poly\|rbf |
| `POST /housing-linear-regression` | `TreeEnsembleService.housing_linear_regression` | — |
| `POST /decision-tree-regression` | `TreeEnsembleService.decision_tree_regression` | `max_depth` |
| `POST /random-forest-regression` | `TreeEnsembleService.random_forest_regression` | `n_estimators`, `max_depth` |
| `POST /classification-algorithm` | `ImageClassificationService.handle_classification_image` | — |
| `POST /logistic-regression-classification` | `LogisticRegressionService.handle_logistic_classification` | — |
| `POST /knn-classification` | `KnnService.handle_knn_classification` | `n_neighbors` (default 5) |
| `GET /health` | — | Healthcheck simple |

Nota importante: **algunas rutas cambiaron de nombre respecto a la versión original** para que reflejen lo que realmente hacen (ver tabla de renombrados en §1.5). Si tenías guardadas peticiones con los nombres viejos (`/tree-regression`, `/random-tree-regression`, `/polynomical-regression`), actualízalas.

Todos los hiperparámetros son opcionales en el body — si mandas `{}` (o nada), se usan los valores por defecto que replican el comportamiento original. Pruébalos desde `/docs` (Swagger) cambiando valores — es la forma más rápida de experimentar con lo que estudias en cada capítulo.

## 1.5 Hallazgos corregidos en esta refactorización

Esta sección documenta **qué estaba mal, por qué, y cómo quedó**. La dejo como historial porque son errores extremadamente comunes al pasar de "notebook" a "servicio real" — vale la pena que recuerdes cómo se ven y cómo se corrigen.

### ✅ Ruta duplicada que hacía KNN inalcanzable
**Antes**: `/logistic-regression-classification` estaba registrada dos veces (regresión logística y KNN), y FastAPI siempre resolvía a la primera — KNN nunca respondía por HTTP aunque el código existiera.
**Ahora**: KNN vive en su propia ruta, `POST /knn-classification`, y además está **implementado de verdad** (antes era un `return "successfully knn classification"` sin lógica). Ver `services/classification/knn_service.py` y la teoría en [07](07-regresion-logistica-y-knn.md).

### ✅ Rutas absolutas hardcodeadas a datasets
**Antes**: cada `pd.read_csv(...)` apuntaba a una ruta fija de tu laptop o tu PC de escritorio (`/mnt/c/Users/Gaylord Carrillo/...`), lo que rompía el proyecto en Docker y en cualquier otra máquina.
**Ahora**: `core/paths.py` calcula la ruta del proyecto con `pathlib` a partir de `__file__`, así que funciona igual sin importar desde dónde se ejecute. **Verificado**: se reconstruyó la imagen Docker y `POST /decision-tree-regression` respondió correctamente dentro del contenedor — antes fallaba.

### ✅ Fuga de datos (data leakage) en el escalado
**Antes**, en la clasificación logística: `X_test = sc_X.fit_transform(X_test)` — el escalador se reajustaba con datos de test.
**Ahora**: `services/shared/social_ads_preprocessing.py` hace `fit_transform` solo en train y `transform` (sin `fit`) en test — la forma correcta, reutilizada también por KNN, que comparte el mismo pipeline de datos.

### ✅ SVR sin escalado de variables
**Antes**: `svr_regression` entrenaba sobre datos sin escalar, pese a que SVR con kernel `rbf` es sensible a la escala.
**Ahora**: `services/regression/svr_service.py` escala `X` **e `Y`** (ver por qué en [05](05-regresion-svr.md)), y des-escala (`inverse_transform`) las predicciones antes de devolverlas. El `R²` subió de ~0.60 (sin escalar) a **~0.98** (escalado) en pruebas locales — la mejora es tan grande que es la evidencia más directa de por qué el escalado importa en este tipo de modelo.

### ✅ Nombres de método/ruta que no coincidían con lo que hacían
| Antes | Qué hacía realmente | Ahora |
|---|---|---|
| `tree_regression()` / `POST /tree-regression` | Regresión **lineal** (sin ningún árbol) sobre `housing.csv` | `housing_linear_regression()` / `POST /housing-linear-regression` |
| `random_tree_regression()` / `POST /random-tree-regression` | Un **árbol de decisión** simple (`DecisionTreeRegressor`) | `decision_tree_regression()` / `POST /decision-tree-regression` |
| `random_forest_regression()` / `POST /random-forest-regression` | Random Forest (ya estaba bien nombrado) | Sin cambios |
| `polynomical_regression()` / `POST /polynomical-regression` | Regresión polinómica (typo en "polynomical") | `polynomical_regression()` (nombre interno sin tocar) / `POST /polynomial-regression` (ruta con ortografía correcta) |

### ✅ Duplicación de ~40 líneas repetidas 3 veces
**Antes**: `tree_regression`, `random_tree_regression` y `random_forest_regression` repetían casi idéntico el bloque de *feature engineering* + imputación + encoding de `housing.csv`.
**Ahora**: `services/shared/housing_preprocessing.py::prepare_housing_dataset()` centraliza esa lógica; los tres servicios de regresión sobre `housing.csv` solo se encargan de instanciar y entrenar su modelo (`_incremental_column_scores` en `tree_ensemble_service.py` factoriza también el loop de selección incremental de columnas).

### ✅ `train_test_split` sin `random_state` en el pipeline de `housing.csv`
**Antes**: cada corrida daba un split distinto, dificultando comparar resultados entre corridas.
**Ahora**: `random_state=42` fijo, igual que en el resto del proyecto — reproducibilidad consistente.

### ✅ Backend de matplotlib interactivo (bug descubierto durante la refactorización, no documentado antes)
Al escribir los tests automatizados, correr varios endpoints que generan gráficos **en el mismo proceso** provocó un *crash* nativo (`Fatal Python error: Aborted`). La causa: matplotlib usaba por defecto el backend interactivo `tkagg` (pensado para mostrar ventanas), inestable en un proceso servidor sin entorno gráfico persistente. Se fuerza `matplotlib.use("Agg")` (backend no interactivo, solo-archivo) al inicio de `main.py`, antes de que cualquier módulo importe `pyplot`. Es la práctica estándar para cualquier servidor que genere gráficos.

### ✅ Endpoints que no devolvían nada útil
Varios métodos solo `print`-eaban las métricas en la consola del servidor y devolvían el dataset crudo o un string genérico (ej. `svr_regression` retornaba `data_dict`, no las predicciones; `random_forest_regression` retornaba el string `"Test random forest regression"`). Ahora todos los endpoints devuelven JSON estructurado con las métricas relevantes (`r2_score`, `rmse`, `precision`, `recall`, `f1_score`, según el caso) — puedes verlas directamente en Swagger sin mirar los logs del servidor.

## 1.6 Pruebas automatizadas (nuevo)

El proyecto no tenía ningún test. Ahora existe `tests/` con `pytest` + `TestClient` de FastAPI:

- `tests/test_health.py` — incluye una prueba de regresión que falla si dos rutas vuelven a compartir el mismo path (el bug de §1.5 no debería poder repetirse sin que un test lo detecte).
- `tests/test_polynomial_regression.py` — valida el endpoint más rápido de probar (dataset sintético, sin I/O externo) y sirve de ejemplo para testear ML: validación de input (`degree=0` debe rechazarse), comportamiento esperado del modelo (mayor `degree` ajusta al menos igual de bien el set de entrenamiento).
- `tests/test_classification.py` — valida regresión logística y confirma que KNN es alcanzable (regression test del bug corregido).

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Ver [10-hoja-de-ruta.md](10-hoja-de-ruta.md) para ideas de qué testear a continuación a medida que agregues técnicas nuevas.

## 1.7 Próximo nivel de arquitectura (para cuando quieras seguir creciendo)

Ideas para cuando el proyecto crezca más:

- **Separar entrenamiento de predicción**: hoy cada request entrena el modelo desde cero. El siguiente paso natural es persistir modelos entrenados con `joblib` y separar `POST /train` de `POST /predict` (ver [10-hoja-de-ruta.md](10-hoja-de-ruta.md)).
- **`sklearn.pipeline.Pipeline` + `ColumnTransformer`**: reemplazaría el patrón manual `fit`/`transform` que usan `svr_service.py` y `social_ads_preprocessing.py`, haciendo estructuralmente imposible repetir el bug de fuga de datos.
- **Un router por bloque temático** cuando agregues clustering/PCA (`routers/clustering.py`, `services/clustering/`), siguiendo el mismo patrón que ya existe.
