# 1. Arquitectura del proyecto

## 1.1 Visión general

El proyecto es una API en **FastAPI** que expone modelos de Machine Learning como endpoints HTTP. Arquitectura en 3 capas:

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

Además existe una cuarta capa, sin relación con ML: **persistencia relacional** (`models.py` + `database.py` + Alembic + PostgreSQL) para un modelo `User`, independiente de los servicios de ML.

## 1.2 Inventario de componentes

```
machine_learning/
├── main.py                     # Crea la app FastAPI, registra routers, exception handlers y matplotlib.use("Agg")
├── core/
│   ├── paths.py                # Rutas absolutas basadas en __file__ (sample_data, results_graphics)
│   ├── exceptions.py           # Excepciones de dominio (InvalidTrainingDataError, UpstreamServiceError)
│   └── security.py             # Hash y verificación de contraseñas con bcrypt (modelo User)
├── schemas.py                  # Contratos Pydantic de entrada (uno por endpoint configurable)
├── database.py / models.py     # SQLAlchemy (no relacionado con ML)
├── routers/
│   ├── regression.py           # 8 endpoints de regresión — capa delgada
│   └── classification.py       # 3 endpoints de clasificación — capa delgada
├── services/
│   ├── shared/
│   │   ├── housing_preprocessing.py       # Pipeline compartido de housing.csv
│   │   ├── social_ads_preprocessing.py    # Pipeline compartido de Social_Network_Ads.csv
│   │   └── plotting.py                    # saved_figure(): guarda y cierra siempre la figura de matplotlib
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

Un archivo = una técnica: para repasar SVR basta con abrir `services/regression/svr_service.py`.

### `machine_learning/core/paths.py`
Centraliza las rutas del proyecto usando `pathlib` y `Path(__file__).resolve()`. Funciona sin importar desde qué directorio se lance `uvicorn` y es idéntico corriendo en local o en Docker.

### `machine_learning/services/shared/`
Cada dataset que se reutiliza en más de una técnica (`housing.csv`, `Social_Network_Ads.csv`) tiene aquí su función de preparación de datos compartida, para no repetir el mismo bloque de limpieza y codificación en cada service. `plotting.py` concentra el ciclo de vida de las figuras de matplotlib.

## 1.3 Flujo de una petición típica

Ejemplo con `POST /v1/linear-regression`:

1. `routers/regression.py` recibe el POST, valida el body contra `RegressionSchema` (Pydantic).
2. Instancia `LinearRegressionService()` y llama a `regression_linear_model(request=request)`.
3. `services/regression/linear_regression_service.py`:
   - Carga `Advertising.csv` vía `core/paths.py`.
   - Arma `X` (la columna elegida) e `Y` (Sales).
   - Divide en train/test, entrena `LinearRegression`, calcula `RMSE`/`R²`.
   - Genera y guarda un gráfico en `results_graphics/`.
   - Devuelve `{"predictions": [...], "rmse": ..., "r2_score": ..., "plot_file": ...}`.
4. FastAPI serializa la respuesta a JSON.

## 1.4 Endpoints disponibles

| Método/Ruta | Service | Hiperparámetros configurables (body) |
|---|---|---|
| `POST /v1/machine-learning` | `LinearRegressionService.handle_user_query` | — |
| `POST /v1/linear-regression` | `LinearRegressionService.regression_linear_model` | `column_name`: TV\|Radio\|Newspaper |
| `POST /v1/multi-linear-regression` | `LinearRegressionService.regression_multi_linear_model` | — |
| `POST /v1/polynomial-regression` | `PolynomialRegressionService.polynomical_regression` | `degree` (1-10, default 4) |
| `POST /v1/svr-regression` | `SvrRegressionService.svr_regression` | `kernel`: linear\|poly\|rbf |
| `POST /v1/housing-linear-regression` | `TreeEnsembleService.housing_linear_regression` | — |
| `POST /v1/decision-tree-regression` | `TreeEnsembleService.decision_tree_regression` | `max_depth` |
| `POST /v1/random-forest-regression` | `TreeEnsembleService.random_forest_regression` | `n_estimators`, `max_depth` |
| `POST /v1/classification-algorithm` | `ImageClassificationService.handle_classification_image` | — |
| `POST /v1/logistic-regression-classification` | `LogisticRegressionService.handle_logistic_classification` | — |
| `POST /v1/knn-classification` | `KnnService.handle_knn_classification` | `n_neighbors` (1-50, default 5) |
| `GET /health` | — | Healthcheck simple (sin versionar: es un endpoint de infraestructura, no parte del contrato de la API) |

Todas las rutas de ML viven bajo el prefijo `/v1` (`APIRouter(prefix="/v1", ...)` en `routers/regression.py` y `routers/classification.py`).

Todos los hiperparámetros son opcionales en el body: si se envía `{}` (o nada), se usan los valores por defecto. Se pueden probar desde `/docs` (Swagger) cambiando valores, que es la forma más rápida de experimentar con lo que se explica en cada capítulo.

## 1.5 Decisiones de diseño

### Rutas de datasets relativas al proyecto
`core/paths.py` calcula la ruta del proyecto con `pathlib` a partir de `__file__`, de modo que ningún `pd.read_csv(...)` depende de una ruta absoluta de una máquina concreta. Esto permite ejecutar el proyecto igual en local y dentro del contenedor Docker.

### Sin fuga de datos (data leakage) en el escalado
`services/shared/social_ads_preprocessing.py` hace `fit_transform` solo sobre train y `transform` (sin `fit`) sobre test. La regresión logística y KNN comparten esta misma preparación.

### SVR con variables escaladas
`services/regression/svr_service.py` escala `X` **e `Y`** (ver por qué en [05](05-regresion-svr.md)) y des-escala las predicciones con `inverse_transform` antes de devolverlas. SVR con kernel `rbf` es sensible a la escala: en pruebas locales, el `R²` fue de ~0.60 sin escalar frente a ~0.98 escalando.

### Nombres de ruta y de método acordes al modelo
Cada nombre describe lo que entrena realmente. `housing-linear-regression` es una regresión **lineal** sobre `housing.csv`, `decision-tree-regression` entrena un `DecisionTreeRegressor` y `random-forest-regression` un `RandomForestRegressor`. La ruta `polynomial-regression` conserva la ortografía correcta, aunque el método interno se llama `polynomical_regression()`.

### Preprocesamiento compartido
`services/shared/housing_preprocessing.py::prepare_housing_dataset()` centraliza la ingeniería de atributos, la imputación y el encoding de `housing.csv`. Los tres servicios sobre ese dataset solo instancian y entrenan su modelo. `_incremental_column_scores` en `tree_ensemble_service.py` factoriza el bucle de selección incremental de columnas.

### Reproducibilidad
Todos los `train_test_split` usan `random_state` fijo (42 en regresión, 0 en clasificación sobre `Social_Network_Ads.csv`), de modo que los resultados son comparables entre corridas.

### Backend de matplotlib no interactivo
`main.py` ejecuta `matplotlib.use("Agg")` antes de que cualquier módulo importe `pyplot`. El backend por defecto (`tkagg`) está pensado para mostrar ventanas y es inestable en un proceso servidor sin entorno gráfico: ejecutar varios endpoints con gráficos en el mismo proceso puede terminar en un *crash* nativo (`Fatal Python error: Aborted`). `Agg` solo escribe archivos y es la práctica estándar para servidores que generan gráficos.

### Respuestas JSON estructuradas
Todos los endpoints devuelven JSON con las métricas relevantes (`r2_score`, `rmse`, `precision`, `recall`, `f1_score`, según el caso), visibles directamente en Swagger sin consultar los logs del servidor.

### Manejo de errores centralizado
Los services no construyen `HTTPException`: levantan excepciones de dominio de `core/exceptions.py` (`InvalidTrainingDataError` → 422, `UpstreamServiceError` → 503), que `main.py` traduce a respuestas HTTP. Cualquier error no previsto termina en un handler genérico que responde 500 sin exponer detalles internos.

## 1.6 Pruebas automatizadas

El directorio `tests/` usa `pytest` + `TestClient` de FastAPI:

- `tests/test_health.py` — healthcheck y una prueba que falla si dos rutas comparten el mismo `(método, path)`, lo que dejaría un endpoint inalcanzable.
- `tests/test_polynomial_regression.py` — valida el endpoint más rápido de probar (dataset sintético, sin I/O externo) y sirve de ejemplo para testear ML: validación de input (`degree=0` debe rechazarse) y comportamiento esperado del modelo (mayor `degree` ajusta al menos igual de bien el set de entrenamiento).
- `tests/test_regression.py` — pruebas de los endpoints de regresión y de las funciones de entrenamiento puras, sin disco ni gráficos.
- `tests/test_classification.py` — regresión logística con métricas exactas, y KNN (alcanzable y con validación de `n_neighbors`).
- `tests/test_image_classification.py` — timeout de la descarga de MNIST, sin usar la red.
- `tests/test_plotting.py` — nombres únicos de gráficos, concurrencia y limpieza de archivos temporales.
- `tests/test_exceptions.py` — mapeo de excepciones de dominio a códigos HTTP.
- `tests/test_security.py` — hash de contraseñas con bcrypt.

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Ver [10-hoja-de-ruta.md](10-hoja-de-ruta.md) para ideas de qué testear a medida que se agreguen técnicas nuevas.

## 1.7 Próximo nivel de arquitectura

Ideas para cuando el proyecto crezca:

- **Separar entrenamiento de predicción**: hoy cada request entrena el modelo desde cero. El siguiente paso natural es persistir modelos entrenados con `joblib` y separar `POST /train` de `POST /predict` (ver [10-hoja-de-ruta.md](10-hoja-de-ruta.md)).
- **`sklearn.pipeline.Pipeline` + `ColumnTransformer`**: reemplazaría el patrón manual `fit`/`transform` que usan `svr_service.py` y `social_ads_preprocessing.py`, haciendo estructuralmente imposible la fuga de datos.
- **Un router por bloque temático** al agregar clustering/PCA (`routers/clustering.py`, `services/clustering/`), siguiendo el mismo patrón que ya existe.
