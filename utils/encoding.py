"""
encoding.py

Utilities for encoding and decoding categorical labels.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_labels(
    dataframe: pd.DataFrame,
    column: str,
):
    """
    Encode categorical labels using LabelEncoder.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.

    column : str
        Column containing class labels.

    Returns
    -------
    tuple
        Encoded dataframe and fitted LabelEncoder.
    """

    dataframe = dataframe.copy()
    encoder = LabelEncoder()

    dataframe["label"] = encoder.fit_transform(
        dataframe[column]
    )

    return dataframe, encoder


def decode_labels(
    labels,
    encoder: LabelEncoder
):
    """
    Decode integer labels back to text labels.
    """

    return encoder.inverse_transform(labels)

def save_encoder(
    encoder: LabelEncoder,
    filepath: Path
):
    """
    Save a fitted LabelEncoder.
    """

    joblib.dump(
        encoder, filepath)


def load_encoder(
    filepath: Path,
):
    """
    Load a saved LabelEncoder.
    """

    return joblib.load(filepath)

