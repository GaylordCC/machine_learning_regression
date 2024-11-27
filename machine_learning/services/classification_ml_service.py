from fastapi import HTTPException

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


class ClassificationAlgorithmService:
    def __init__(self) -> None:
        pass

    def handle_classification_image(
        self
    ):
        mnist = fetch_openml('mnist_784', version=1)
        print(mnist.keys())
        return "ok"
