from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"

def resolve_split_dir(split_name: str) -> Path:
    """Resolve a data split folder while supporting legacy `data/raw` layout."""

    direct_split_dir = DATA_DIR / split_name
    if direct_split_dir.exists():
        return direct_split_dir
    return RAW_DATA_DIR / split_name


TRAIN_DIR = resolve_split_dir("train")
VAL_DIR = resolve_split_dir("val")

IMAGE_SIZE = 224
NUM_CLASSES = 4
RANDOM_SEED = 42
CLASS_NAMES = ("bridge", "freeway", "overpass", "railway")
