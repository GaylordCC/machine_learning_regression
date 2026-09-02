"""Domain-level exceptions services raise instead of building HTTPException by
hand. Global handlers in main.py map these to HTTP responses -- keeps services
free of any FastAPI/HTTP-specific code, and keeps the error-to-status mapping
in one place instead of repeated per service method.
"""


class InvalidTrainingDataError(Exception):
    """Request passed schema validation, but is incompatible with the model or
    dataset at training time (e.g. n_neighbors greater than available samples)."""


class UpstreamServiceError(Exception):
    """A third-party service this endpoint depends on (e.g. OpenML) failed or timed out."""
