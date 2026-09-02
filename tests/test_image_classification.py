"""Unit test for the fetch_openml timeout wrapper.

Doesn't hit the network (the full endpoint needs internet + is slow, kept out
of the suite deliberately -- see documentacion/08). Mocks fetch_openml to
verify the specific property that matters: the global socket timeout is set
during the call and restored afterward, even when the call fails.
"""
import socket
from unittest.mock import patch

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
