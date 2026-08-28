import matplotlib
matplotlib.use("Agg")  # non-interactive backend: a server has no display, and the default
                        # backend can crash the process when it tries to use one (see tests/).

from fastapi import FastAPI

from .routers import regression, classification

app = FastAPI(
    title="Machine Learning Study API",
    description="API de estudio con endpoints de regresion y clasificacion basados en scikit-learn.",
)

app.include_router(regression.router)
app.include_router(classification.router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
