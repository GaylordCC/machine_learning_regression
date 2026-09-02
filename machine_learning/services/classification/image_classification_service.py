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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from ...core.exceptions import UpstreamServiceError
from ..shared.plotting import saved_figure

# fetch_openml has no timeout parameter in its public API (verified: only
# n_retries/delay, which bound retry count but not per-attempt wall time).
# socket.setdefaulttimeout is the only real lever, but it's process-global,
# not thread-local -- a concurrent request using sockets elsewhere in the
# process during this window would also be bound by it. Acceptable trade-off
# for a low-concurrency study API; would need a real per-call timeout
# mechanism (e.g. a subprocess or asyncio.wait_for around a thread) before
# this app ever runs under real concurrent load.
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
        """Fit SGDClassifier ("is this a 5?") + cross-validate. Pure: no I/O,
        no plotting."""
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

    def handle_classification_image(self):
        mnist = self._fetch_mnist()
        X, Y = mnist["data"], mnist["target"]
        Y = Y.astype(np.uint8)

        digit = X.to_numpy()[0]
        fig = saved_figure("plot_classification.png")
        with fig as plt:
            plt.imshow(digit.reshape(28, 28), cmap="binary")

        return {**self._train_digit_classifier(X, Y), "plot_file": fig.filename}
