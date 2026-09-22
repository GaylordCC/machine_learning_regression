"""Unit test for the fetch_openml timeout wrapper.

Doesn't hit the network (the full endpoint needs internet + is slow, kept out
of the suite deliberately -- see documentacion/08). Mocks fetch_openml to
verify the specific property that matters: the global socket timeout is set
during the call and restored afterward, even when the call fails.
"""
import socket
from unittest.mock import patch

import numpy as np
import pytest

from machine_learning.core.exceptions import UpstreamServiceError
from machine_learning.services.classification.image_classification_service import (
    ImageClassificationService,
    OPENML_FETCH_TIMEOUT_SECONDS,
)


def test_openml_fetch_sets_timeout_during_call_and_restores_it_after_failure():
    original_timeout = socket.getdefaulttimeout()
    observed_timeout_during_call = None

    def fake_fetch_openml(*args, **kwargs):
        nonlocal observed_timeout_during_call
        observed_timeout_during_call = socket.getdefaulttimeout()
        raise TimeoutError("simulated network timeout")

    with patch(
        "machine_learning.services.classification.image_classification_service.fetch_openml",
        side_effect=fake_fetch_openml,
    ):
        with pytest.raises(UpstreamServiceError):
            ImageClassificationService().handle_classification_image()

    assert observed_timeout_during_call == OPENML_FETCH_TIMEOUT_SECONDS
    assert socket.getdefaulttimeout() == original_timeout


def test_digit_classifier_scores_both_cross_validation_and_the_test_split():
    """MNIST's 60,000/10,000 split is positional, so a synthetic array with the
    same row count exercises the real code path without any download. A row is
    a "5" when its first feature is large."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60_300, 3))
    Y = np.where(X[:, 0] > 1, 5, 0).astype(np.uint8)

    result = ImageClassificationService()._train_digit_classifier(X, Y)

    assert len(result["cross_val_accuracy"]) == 3
    for key in ("precision", "recall", "f1_score", "test_precision", "test_recall", "test_f1_score"):
        assert 0 <= result[key] <= 1, key
    assert result["test_roc_auc"] > 0.9
