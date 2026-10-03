from pathlib import Path

SEED = 42
OPENML_ID = 1597
TARGET = "Class"

# Business cost assumption: a missed fraud costs 100x a false alert.
C_FN = 100.0
C_FP = 1.0

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
