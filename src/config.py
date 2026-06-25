from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LEGACY_ASSIGNED_DATASET_DIR = DATA_DIR / "set 12"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

def resolve_split_dir(split_name: str, fallback: Path | None = None) -> Path:
    """Resolve a data split folder with canonical raw layout preference.

    New code should use TRAIN_DIR or VAL_DIR directly. This resolver is retained
    for archived scripts that need to tolerate older local folder names.
    """

    raw_split_dir = RAW_DATA_DIR / split_name
    if raw_split_dir.exists():
        return raw_split_dir

    direct_split_dir = DATA_DIR / split_name
    if direct_split_dir.exists():
        return direct_split_dir

    if fallback is not None:
        return fallback

    return raw_split_dir


TRAIN_DIR = RAW_DATA_DIR / "train"
VAL_DIR = RAW_DATA_DIR / "val"
# Deprecated alias retained for archived scripts/manifests that predate the
# canonical `data/raw/train` layout. New code should use TRAIN_DIR.
ASSIGNED_DATASET_DIR = LEGACY_ASSIGNED_DATASET_DIR
SPLIT_MANIFEST_PATH = TABLES_DIR / "split_manifest.csv"

IMAGE_SIZE = 224
NUM_CLASSES = 4
RANDOM_SEED = 42
CLASS_NAMES = ("bridge", "freeway", "overpass", "railway")
