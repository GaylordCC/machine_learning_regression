from fastapi import FastAPI
from .routers import main_ml, classification_ml

# Create an instance of the FastAPI app
app = FastAPI()

# Include the router
app.include_router(main_ml.router)
app.include_router(classification_ml.router)