"""
config.py

Central configuration file for project directories.
"""

from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# =============================================================================
# MODEL & ARTIFACT DIRECTORIES
# =============================================================================

MODEL_DIR = ROOT_DIR / "models"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
ENCODER_DIR = ARTIFACT_DIR / "encoders"
TRANSFORMER_DIR = ARTIFACT_DIR / "transformers"
VECTORIZER_DIR = ARTIFACT_DIR / "vectorizers"
SCALER_DIR = ARTIFACT_DIR / "scalers"

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

from utils.helpers import create_directory

DIRECTORIES = [
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    ARTIFACT_DIR,
    ENCODER_DIR,
    TRANSFORMER_DIR,
    VECTORIZER_DIR,
    SCALER_DIR,
    OUTPUT_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]

for directory in DIRECTORIES:
    create_directory(directory)