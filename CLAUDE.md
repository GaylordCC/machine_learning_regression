# CLAUDE.md

Guía para Claude Code al trabajar en este repositorio.

## Descripción del proyecto

API en **FastAPI** que expone endpoints de Machine Learning (regresión y clasificación, basados en scikit-learn) y persiste datos en **PostgreSQL** mediante **SQLAlchemy** y **Alembic**. Ver `README.md` para los comandos de setup, Docker y migraciones.

## Arquitectura

- `machine_learning/main.py` — punto de entrada de FastAPI; registra los routers.
- `machine_learning/routers/` — define los endpoints HTTP (`main_ml.py` para regresión, `classification_ml.py` para clasificación). Los routers son delgados: delegan toda la lógica a los servicios.
- `machine_learning/services/` — contiene la lógica de negocio y los algoritmos de ML (`main_ml_service.py`, `classification_ml_service.py`).
- `machine_learning/models.py` — modelos ORM de SQLAlchemy.
- `machine_learning/schemas.py` — esquemas Pydantic para request/response.
- `machine_learning/database.py` — engine, sesión y `get_db()` (lee `POSTGRESQL_CONNECTION_URL` desde `.env`).
- `machine_learning/sample_data/` — datasets CSV usados por los algoritmos de ejemplo.
- `results_graphics/` — gráficos generados por los servicios de ML.
- `alembic/` — migraciones de base de datos.

Patrón: Router → Service → (Modelo/DB o dataset CSV). Cada router instancia su Service correspondiente por request (no hay inyección de dependencias para los servicios de ML).

## Comandos de desarrollo

```bash
# Activar entorno virtual
source venv/bin/activate

# Levantar el servidor con recarga automática
uvicorn machine_learning.main:app --reload

# Nueva migración autogenerada tras cambiar machine_learning/models.py
alembic revision --autogenerate -m "mensaje descriptivo"
alembic upgrade head
```

No hay suite de tests ni linter configurado en este repo todavía.

## Convenciones observadas en el código

- Los endpoints usan `@router.post(...)` incluso cuando no reciben body (patrón existente, mantenerlo salvo que se pida lo contrario).
- Los nombres de las funciones de los endpoints se repiten (`process_request`) dentro de cada archivo de router; es el estilo actual del proyecto.
- Comentarios cortos en inglés explicando el propósito de cada bloque/endpoint.
- Las variables de entorno (conexión a PostgreSQL) se cargan vía `python-dotenv` desde un archivo `.env` en la raíz (no versionado).

## Notas

- La base de datos local de desarrollo se llama `ai_recruitment` (PostgreSQL en WSL/Ubuntu).
- El archivo `.env` contiene credenciales — nunca leerlo/mostrarlo en salidas ni commitearlo.
