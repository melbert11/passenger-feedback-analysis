"""
data_loader.py

Functions for loading and saving datasets used throughout the project.
"""

from pathlib import Path

import pandas as pd

from utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# =============================================================================
# LOAD DATASETS
# =============================================================================

def load_raw_feedback(filename: str = "feedback.csv") -> pd.DataFrame:
    """
    Load the raw feedback dataset.

    Parameters
    ----------
    filename : str, optional
        Name of the CSV file inside data/raw.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset not found:\n{filepath}")

    return pd.read_csv(filepath)


def load_processed_feedback(
    filename: str = "feedback_clean.csv",
) -> pd.DataFrame:
    """
    Load a processed dataset.

    Parameters
    ----------
    filename : str, optional
        Name of the processed CSV.

    Returns
    -------
    pd.DataFrame
    """

    filepath = PROCESSED_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Processed dataset not found:\n{filepath}")

    return pd.read_csv(filepath)


# =============================================================================
# SAVE DATASETS
# =============================================================================

def save_processed_feedback(
    dataframe: pd.DataFrame,
    filename: str = "feedback_clean.csv",
) -> Path:
    """
    Save a processed dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to save.

    filename : str
        Output filename.

    Returns
    -------
    Path
        Path of the saved file.
    """

    filepath = PROCESSED_DATA_DIR / filename
    dataframe.to_csv(filepath, index=False)

    return filepath