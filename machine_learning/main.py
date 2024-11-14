from fastapi import FastAPI
from .routers import main_ml

app = FastAPI()


app.include_router(main_ml.router)