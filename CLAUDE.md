# CLAUDE.md

Guía para Claude Code al trabajar en este repositorio.

## Descripción del proyecto

API en **FastAPI** que expone endpoints de Machine Learning (regresión y clasificación, basados en scikit-learn) y persiste datos en **PostgreSQL** mediante **SQLAlchemy** y **Alembic**. Ver `README.md` para los comandos de setup, Docker y migraciones. Es un proyecto de estudio: cada técnica de ML tiene su capítulo teórico correspondiente en `documentacion/`.

## Arquitectura

- `machine_learning/main.py` — punto de entrada de FastAPI; fuerza el backend `Agg` de matplotlib (headless, sin display) antes de registrar los routers, registra los exception handlers globales (ver `core/exceptions.py`) y expone `GET /health`.
- `machine_learning/core/paths.py` — rutas del proyecto (`sample_data/`, `results_graphics/`) resueltas con `pathlib` a partir de `__file__`. Usar siempre esto en vez de rutas hardcodeadas o relativas al cwd.
- `machine_learning/core/security.py` — `hash_password`/`verify_password` con `bcrypt` directo (no `passlib`: incompatible con bcrypt>=4.1, ver comentario en el archivo). Sin usar todavía por ningún endpoint.
- `machine_learning/core/exceptions.py` — excepciones de dominio (`InvalidTrainingDataError` → 422, `UpstreamServiceError` → 503) que los services levantan en vez de construir `HTTPException` a mano. Mapeadas a respuestas HTTP por handlers globales en `main.py`; cualquier excepción no anticipada cae en un handler catch-all → 500 genérico (el detalle real va solo al log, nunca al cliente). Los services ya no deben tener `try/except Exception` genérico — solo capturan un error específico si hay un caso de negocio real que traducir a `InvalidTrainingDataError`/`UpstreamServiceError` (ver `knn_service.py` e `image_classification_service.py` como ejemplos).
- `machine_learning/routers/` — capa HTTP delgada: `regression.py` (8 endpoints) y `classification.py` (3 endpoints). Delegan toda la lógica a los services.
- `machine_learning/services/regression/` — un archivo por técnica: `linear_regression_service.py` (EDA + simple + múltiple), `polynomial_regression_service.py`, `svr_service.py`, `tree_ensemble_service.py` (lineal/árbol/random forest sobre `housing.csv`). Cada método de endpoint sigue el patrón: cargar datos → `_train*()` (función pura, sin I/O ni plotting, testeable con un DataFrame sintético) → `saved_figure(...)` de `services/shared/plotting.py` para el side-effect del gráfico → retornar métricas.
- `machine_learning/services/classification/` — `logistic_regression_service.py`, `knn_service.py`, `image_classification_service.py` (MNIST). Mismo patrón de separación I/O/train/plot donde aplica (ver `image_classification_service.py`: `_fetch_mnist()` / `_train_digit_classifier()` / plot).
- `machine_learning/services/shared/` — utilidades reutilizadas por varias técnicas: `housing_preprocessing.py` y `social_ads_preprocessing.py` (pipelines de datos), `plotting.py` (`saved_figure()`: guarda y cierra la figura de matplotlib, siempre, incluso si el dibujo falla a medias — evita fugas de figuras entre requests).
- `machine_learning/schemas.py` — todos los schemas Pydantic de request (uno por endpoint con hiperparámetros configurables: `degree`, `kernel`, `max_depth`, `n_estimators`, `n_neighbors`).
- `machine_learning/models.py` / `database.py` — SQLAlchemy para un modelo `User` (columna `hashed_password`, no `password` — ver `core/security.py`), sin relación con los endpoints de ML.
- `machine_learning/sample_data/` — datasets CSV usados por los servicios.
- `results_graphics/` — gráficos generados por los servicios (creado automáticamente si no existe, vía `core/paths.py`).
- `tests/` — pytest + `TestClient` de FastAPI. Correr con `pytest tests/ -v` (requiere `requirements-dev.txt`).
- `alembic/` — migraciones de base de datos (no relacionado con ML).
- `documentacion/` — material de estudio de ML en español, generado a partir del código real del proyecto (teoría + referencias a archivo/técnica). Ver su `README.md` para el índice completo. **Al agregar una técnica nueva a los services, añade también su capítulo correspondiente ahí** y su fila en la tabla de endpoints de `documentacion/01-arquitectura-del-proyecto.md`.

Patrón: Router → Service → (dataset CSV vía `core/paths.py`, o pipeline compartido en `services/shared/`). Cada router instancia su Service correspondiente por request (no hay inyección de dependencias para los servicios de ML). Todos los endpoints devuelven JSON estructurado con las métricas relevantes (`r2_score`, `rmse`, `precision`, `recall`, `f1_score`, según el caso). Los services no atrapan excepciones genéricas ni construyen `HTTPException`: dejan subir bugs reales (→ 500 vía handler global) y solo levantan las excepciones de dominio de `core/exceptions.py` para casos de negocio recuperables.

## Comandos de desarrollo

```bash
# Activar entorno virtual
source venv/bin/activate

# Levantar el servidor con recarga automática
uvicorn machine_learning.main:app --reload

# Correr los tests
pip install -r requirements-dev.txt
pytest tests/ -v

# Nueva migración autogenerada tras cambiar machine_learning/models.py
alembic revision --autogenerate -m "mensaje descriptivo"
alembic upgrade head
```

No hay linter configurado en este repo todavía.

## Convenciones del código

- Un service por técnica de ML, bajo `services/regression/` o `services/classification/`. Si una técnica nueva comparte pipeline de datos con otra existente, extraer a `services/shared/`.
- Nunca usar rutas absolutas ni relativas al cwd para leer CSVs — siempre `core/paths.py::sample_data_path()`.
- Cualquier `fit`/`fit_transform` de un scaler/encoder va solo sobre datos de train; test siempre usa `transform` (evitar data leakage — ver `documentacion/02` y `documentacion/07`).
- Los hiperparámetros expuestos al usuario (degree, kernel, max_depth, n_estimators, n_neighbors...) se declaran como schemas Pydantic en `schemas.py` con defaults que preservan el comportamiento histórico del proyecto.
- Comentarios cortos en inglés explicando el "por qué", no el "qué" (la teoría detallada vive en `documentacion/`, no en los comentarios del código).
- Las variables de entorno (conexión a PostgreSQL) se cargan vía `python-dotenv` desde un archivo `.env` en la raíz (no versionado).

## Notas

- La base de datos local de desarrollo se llama `ai_recruitment` (PostgreSQL en WSL/Ubuntu).
- El archivo `.env` contiene credenciales — nunca leerlo/mostrarlo en salidas ni commitearlo.
- `POST /classification-algorithm` (MNIST) descarga datos de OpenML la primera vez que se llama — requiere internet y tarda; no es apto para tests rápidos (ver `tests/`, que evita ese endpoint). La llamada a `fetch_openml` tiene un timeout de 30s vía `socket.setdefaulttimeout` (global al proceso, no thread-safe — ver comentario en `image_classification_service.py`; `fetch_openml` no expone un parámetro de timeout propio).
- Dependencias base actualizadas (fastapi 0.141.1, starlette 1.6.0, python-dotenv 1.2.3) para resolver CVEs conocidos — verificado con `pip-audit` y suite completa antes de fijar versiones.
