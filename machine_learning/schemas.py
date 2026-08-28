from typing import Optional

from pydantic import BaseModel, Field
from enum import Enum


class MediaType(str, Enum):
    TV = "TV"
    Radio = "Radio"
    Newspaper = "Newspaper"


class RegressionSchema(BaseModel):
    column_name: MediaType


class PolynomialRegressionSchema(BaseModel):
    degree: int = Field(default=4, ge=1, le=10, description="Grado del polinomio. Sube el valor para ver overfitting.")


class SvrKernel(str, Enum):
    linear = "linear"
    poly = "poly"
    rbf = "rbf"


class SvrRegressionSchema(BaseModel):
    kernel: SvrKernel = SvrKernel.rbf


class TreeRegressionSchema(BaseModel):
    max_depth: Optional[int] = Field(default=None, ge=1, description="Profundidad maxima del arbol. None = sin limite (riesgo de overfitting).")


class RandomForestRegressionSchema(BaseModel):
    n_estimators: int = Field(default=100, ge=1, le=500)
    max_depth: Optional[int] = Field(default=None, ge=1)


class KnnClassificationSchema(BaseModel):
    n_neighbors: int = Field(default=5, ge=1, le=50)
