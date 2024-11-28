from fastapi import HTTPException

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

import os

class ClassificationAlgorithmService:
    def __init__(self) -> None:
        pass

    def handle_classification_image(
        self
    ):
        mnist = fetch_openml('mnist_784', version=1)
        print(mnist.keys())

        # Get mnist data and target
        X, Y = mnist['data'], mnist['target']
        print(X.shape)
        print(Y.shape)
        # Convert X data into numpy array
        digit = X.to_numpy()[0]
        digit_image = digit.reshape(28,28)

        # Generating a plot and storage it
        results_graphics_path = 'results_graphics'
        filenames = []
        plt.imshow(digit_image, cmap='binary')
        file_path = os.path.join(results_graphics_path, f"plot_classification.png")
        plt.savefig(file_path)
        plt.close()  # Cierra el gráfico para liberar memoria
        filenames.append(file_path)

        # Verifying the value teste above
        Y = Y.astype(np.uint8)
        print(Y[0])
        return "ok"
