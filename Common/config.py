from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIXTURE_DIR = ROOT / "tests" / "fixtures"

WCS_DATA_PAGE_URL = "https://linguistics.berkeley.edu/wcs/data.html"
WCS_MAPPING_URL = "https://linguistics.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt"
MAPPING_NAME = "cnum-vhcm-lab-new.txt"
WCS_USE_FIXTURE = os.environ.get("WCS_USE_FIXTURE", "0") == "1"

SEED = 20260821
MIN_SPEAKER_FRACTION = 0.20
MIN_CHIPS_PER_TERM = 3
EXPECTED_WCS_LANGUAGES = 110
EXPECTED_WCS_CHIPS = 330

for p in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR / "project1", RESULTS_DIR / "project2"]:
    p.mkdir(parents=True, exist_ok=True)
