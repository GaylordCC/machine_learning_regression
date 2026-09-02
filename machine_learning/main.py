import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: a server has no display, and the default
                        # backend can crash the process when it tries to use one (see tests/).

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .core.exceptions import InvalidTrainingDataError, UpstreamServiceError
from .routers import regression, classification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("machine_learning")

app = FastAPI(
    title="Machine Learning Study API",
    description="API de estudio con endpoints de regresion y clasificacion basados en scikit-learn.",
)

app.include_router(regression.router)
app.include_router(classification.router)


@app.exception_handler(InvalidTrainingDataError)
def invalid_training_data_handler(request: Request, exc: InvalidTrainingDataError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(UpstreamServiceError)
def upstream_service_handler(request: Request, exc: UpstreamServiceError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak str(exc) to the client -- it can contain internal paths/data.
    # Full detail goes to the log; the response stays generic. Registering a
    # handler for the bare Exception class makes Starlette run it as the
    # ServerErrorMiddleware handler via a worker thread, where sys.exc_info()
    # isn't populated -- pass exc_info explicitly instead of logger.exception().
    logger.error("Unhandled error handling %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
