from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DIR = RAW_DATA_DIR / "train"
VAL_DIR = RAW_DATA_DIR / "val"

IMAGE_SIZE = 224
NUM_CLASSES = 4
RANDOM_SEED = 42

