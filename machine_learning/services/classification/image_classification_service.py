"""Binary MNIST classifier ("is this digit a 5?") with cross-validation.

See documentacion/08-clasificacion-mnist-y-metricas.md for the theory.
Downloads MNIST from OpenML on first run (requires internet access);
scikit-learn caches it locally afterwards.
"""
from fastapi import HTTPException
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from ...core.paths import results_graphics_path


class ImageClassificationService:
    def handle_classification_image(self):
        try:
            mnist = fetch_openml("mnist_784", version=1)
            X, Y = mnist["data"], mnist["target"]
            Y = Y.astype(np.uint8)

            digit = X.to_numpy()[0]
            plt.imshow(digit.reshape(28, 28), cmap="binary")
            plt.savefig(results_graphics_path("plot_classification.png"))
            plt.close()

            X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
            Y_train_5 = Y_train == 5

            sgd_classifier = SGDClassifier(random_state=42)
            sgd_classifier.fit(X_train, Y_train_5)

            cv_accuracy = cross_val_score(sgd_classifier, X_train, Y_train_5, cv=3, scoring="accuracy")

            Y_train_predict = cross_val_predict(sgd_classifier, X_train, Y_train_5, cv=3)

            return {
                "cross_val_accuracy": cv_accuracy.tolist(),
                "confusion_matrix": confusion_matrix(Y_train_5, Y_train_predict).tolist(),
                "precision": precision_score(Y_train_5, Y_train_predict),
                "recall": recall_score(Y_train_5, Y_train_predict),
                "f1_score": f1_score(Y_train_5, Y_train_predict),
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"image classification: {str(e)}")
