"""Binary MNIST classifier ("is this digit a 5?") with cross-validation.

See documentacion/08-clasificacion-mnist-y-metricas.md for the theory.
Downloads MNIST from OpenML on first run (requires internet access);
scikit-learn caches it locally afterwards.
"""
import socket
from urllib.error import URLError

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from ...core.exceptions import UpstreamServiceError
from ..shared.plotting import saved_figure

# fetch_openml has no timeout parameter (only n_retries/delay, which bound the
# retry count, not the time per attempt), so socket.setdefaulttimeout is the
# only lever. It is process-global, not thread-local: a concurrent request
# using sockets during this window is also bound by it. Acceptable for a
# low-concurrency study API; under real concurrent load it needs a per-call
# mechanism (e.g. a subprocess, or asyncio.wait_for around a thread).
OPENML_FETCH_TIMEOUT_SECONDS = 30


class ImageClassificationService:
    def _fetch_mnist(self):
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(OPENML_FETCH_TIMEOUT_SECONDS)
        try:
            return fetch_openml("mnist_784", version=1)
        except (URLError, OSError) as e:
            # Covers socket.timeout/TimeoutError/ConnectionError (all OSError
            # subclasses) and urllib's own URLError -- a network problem, not
            # something the caller's request caused.
            raise UpstreamServiceError(f"Could not fetch MNIST from OpenML: {e}") from e
        finally:
            socket.setdefaulttimeout(previous_timeout)

    def _train_digit_classifier(self, X, Y) -> dict:
        """Fit SGDClassifier ("is this a 5?"), cross-validate on the train split
        and score on the held-out test split. Pure: no I/O, no plotting.

        The unprefixed metrics come from cross-validation on train (each
        prediction made by a fold that did not train on it); the `test_*`
        metrics come from MNIST's standard 10,000-image test split.
        """
        X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
        Y_train_5 = Y_train == 5
        Y_test_5 = Y_test == 5

        sgd_classifier = SGDClassifier(random_state=42)
        sgd_classifier.fit(X_train, Y_train_5)

        cv_accuracy = cross_val_score(sgd_classifier, X_train, Y_train_5, cv=3, scoring="accuracy")
        Y_train_predict = cross_val_predict(sgd_classifier, X_train, Y_train_5, cv=3)

        Y_test_predict = sgd_classifier.predict(X_test)
        # SGDClassifier has no predict_proba by default; the decision score is
        # enough for ROC-AUC, which only needs the ranking.
        Y_test_score = sgd_classifier.decision_function(X_test)

        return {
            "cross_val_accuracy": cv_accuracy.tolist(),
            "confusion_matrix": confusion_matrix(Y_train_5, Y_train_predict).tolist(),
            "precision": precision_score(Y_train_5, Y_train_predict),
            "recall": recall_score(Y_train_5, Y_train_predict),
            "f1_score": f1_score(Y_train_5, Y_train_predict),
            "test_precision": precision_score(Y_test_5, Y_test_predict),
            "test_recall": recall_score(Y_test_5, Y_test_predict),
            "test_f1_score": f1_score(Y_test_5, Y_test_predict),
            "test_roc_auc": roc_auc_score(Y_test_5, Y_test_score),
        }

    def handle_classification_image(self):
        mnist = self._fetch_mnist()
        X, Y = mnist["data"], mnist["target"]
        Y = Y.astype(np.uint8)

        digit = X.to_numpy()[0]
        fig = saved_figure("plot_classification.png")
        with fig as plt:
            plt.imshow(digit.reshape(28, 28), cmap="binary")

        return {**self._train_digit_classifier(X, Y), "plot_file": fig.filename}
