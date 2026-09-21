"""Shared matplotlib figure lifecycle for the plotting services.

The figure is always closed, even if drawing raises partway through, because
matplotlib keeps every open figure in memory until it's closed. It also keeps
the training/scoring logic separate from the plotting side effect: services
call this only around the draw+save step.

matplotlib.pyplot's "current figure" is global, process-wide mutable state --
not thread-local. FastAPI runs each sync route in a worker thread, so
concurrent requests to plotting endpoints would drive it at once, and one
thread's plt.close("all") could tear down a figure another thread is still
drawing on (leaving a blank plot). A lock around the whole draw+save+close
cycle prevents this.

Each call writes its own uniquely-named file instead of overwriting a fixed
"latest run" name, so a caller can trace which plot belongs to which request
-- see `filename` below and the `plot_file` key each service adds to its
response. Callers should build `base_filename` from the request's own
parameters where they exist (e.g. `f"plotregression_{column_name}.png"`), not
just the technique name; otherwise two different requests to the same
endpoint are indistinguishable by filename alone, unique suffix or not.
"""
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import date

import matplotlib.pyplot as plt

from ...core.paths import results_graphics_path

# Serializes every saved_figure() call across the whole process. Global on
# purpose: the race is in pyplot's shared state, not per-file, so locking
# per-filename would not stop two different plots from corrupting each other.
_PYPLOT_LOCK = threading.Lock()


def _unique_path(base_filename: str):
    base = results_graphics_path(base_filename)
    stamp = date.today().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:8]
    return base.with_name(f"{base.stem}_{stamp}_{suffix}{base.suffix}")


class _SavedFigure:
    """Context manager: yields pyplot to draw on, and exposes `.filename`
    (the actual unique name written under results_graphics/) once the
    `with` block completes successfully -- `None` if it raised instead."""

    def __init__(self, base_filename: str):
        self._base_filename = base_filename
        self.filename: str | None = None

    def __enter__(self):
        _PYPLOT_LOCK.acquire()
        return plt

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                final_path = _unique_path(self._base_filename)
                # Keep the real suffix (.png) so savefig() still infers the
                # format from the extension -- a bare ".tmp" suffix raises.
                tmp_path = final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")
                try:
                    plt.savefig(tmp_path)
                    os.replace(tmp_path, final_path)
                    self.filename = final_path.name
                except BaseException:
                    tmp_path.unlink(missing_ok=True)
                    raise
        finally:
            plt.close("all")
            _PYPLOT_LOCK.release()
        return False  # never swallow an exception raised inside the `with` block


def saved_figure(base_filename: str) -> _SavedFigure:
    """`with saved_figure("plotregression_TV.png") as plt: ...` -- draw on
    `plt` as usual; after the block, use the returned object's `.filename`
    for the actual unique name that was written to disk."""
    return _SavedFigure(base_filename)
