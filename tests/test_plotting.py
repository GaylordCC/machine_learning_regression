"""Tests for services/shared/plotting.py: the per-request unique filename,
and the regression test for the concurrency bug that made the lock necessary
in the first place.

matplotlib.pyplot keeps its "current figure" as global, process-wide state,
not thread-local. Reproduced empirically (before the lock existed): two
threads drawing distinguishable content concurrently through saved_figure()
could end up with one thread's plot completely blank, because the other
thread's plt.close("all") tore down the figure it was still drawing on.
"""
import re
import threading

import numpy as np
from PIL import Image

from machine_learning.core.paths import RESULTS_GRAPHICS_DIR
from machine_learning.services.shared.plotting import saved_figure

N_THREADS = 6


def _draw(i: int, results: dict, errors: list) -> None:
    color = "r" if i % 2 == 0 else "b"
    try:
        fig = saved_figure(f"test_race_{i}.png")
        with fig as plt:
            plt.plot([0, 1], [i, i], f"{color}-", linewidth=10)
        results[i] = fig.filename
    except Exception as e:
        errors.append((i, e))


def _dominant_color(path) -> str:
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    red_px = ((r > 180) & (g < 100) & (b < 100)).sum()
    blue_px = ((b > 180) & (r < 100) & (g < 100)).sum()
    if red_px > blue_px:
        return "red"
    if blue_px > red_px:
        return "blue"
    return "none"


def test_concurrent_saved_figure_calls_do_not_corrupt_each_others_content():
    results = {}
    errors = []
    threads = [threading.Thread(target=_draw, args=(i, results, errors)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == N_THREADS

    try:
        for i in range(N_THREADS):
            expected = "red" if i % 2 == 0 else "blue"
            path = RESULTS_GRAPHICS_DIR / results[i]
            assert _dominant_color(path) == expected, f"figure {i} was corrupted by a concurrent call"
    finally:
        for filename in results.values():
            (RESULTS_GRAPHICS_DIR / filename).unlink(missing_ok=True)


def test_saved_figure_generates_a_unique_permanent_filename_per_call():
    """Each call gets its own YYYYMMDD + random-suffix filename instead of
    overwriting one fixed name -- two calls with the same base never collide."""
    fig_a = saved_figure("plotregression_TV.png")
    with fig_a as plt:
        plt.plot([0, 1], [0, 1])

    fig_b = saved_figure("plotregression_TV.png")
    with fig_b as plt:
        plt.plot([0, 1], [1, 0])

    try:
        assert fig_a.filename != fig_b.filename
        assert re.match(r"^plotregression_TV_\d{8}_[0-9a-f]{8}\.png$", fig_a.filename)
        assert (RESULTS_GRAPHICS_DIR / fig_a.filename).exists()
        assert (RESULTS_GRAPHICS_DIR / fig_b.filename).exists()
    finally:
        (RESULTS_GRAPHICS_DIR / fig_a.filename).unlink(missing_ok=True)
        (RESULTS_GRAPHICS_DIR / fig_b.filename).unlink(missing_ok=True)


def test_saved_figure_leaves_no_temp_file_behind():
    fig = saved_figure("test_no_leftover_tmp.png")
    with fig as plt:
        plt.plot([0, 1], [0, 1])

    final_path = RESULTS_GRAPHICS_DIR / fig.filename
    try:
        assert final_path.exists()
        assert list(RESULTS_GRAPHICS_DIR.glob("test_no_leftover_tmp*.tmp*")) == []
    finally:
        final_path.unlink(missing_ok=True)


def test_saved_figure_filename_is_none_when_drawing_raises():
    fig = saved_figure("test_never_saved.png")
    try:
        with fig as plt:
            plt.plot([0, 1], [0, 1])
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert fig.filename is None
    assert list(RESULTS_GRAPHICS_DIR.glob("test_never_saved*")) == []
