from pathlib import Path

# Project root: machine_learning_v01/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

SAMPLE_DATA_DIR = BASE_DIR / "machine_learning" / "sample_data"
RESULTS_GRAPHICS_DIR = BASE_DIR / "results_graphics"

RESULTS_GRAPHICS_DIR.mkdir(exist_ok=True)


def sample_data_path(filename: str) -> Path:
    return SAMPLE_DATA_DIR / filename


def results_graphics_path(filename: str) -> Path:
    return RESULTS_GRAPHICS_DIR / filename
