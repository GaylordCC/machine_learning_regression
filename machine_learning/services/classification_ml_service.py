from fastapi import HTTPException

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

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

        X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
        Y_train_5 = (Y_train == 5)
        Y_test_5 = (Y_test == 5)

        # SGDC Classifier (Algorithm)
        sgd_classifier = SGDClassifier(random_state=42)
        sgd_classifier.fit(X_train, Y_train_5)

        # 
        print(sgd_classifier.predict([digit]))

        # Medir el rendiemiento del modelo
        print(cross_val_score(sgd_classifier, X_train, Y_train_5, cv=3, scoring='accuracy'))

        return "ok"
