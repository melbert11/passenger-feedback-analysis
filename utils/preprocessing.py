"""
preprocessing.py

Reusable text preprocessing functions for driver feedback analysis.
"""

import re
import html
import string

import pandas as pd

def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    """

    return re.sub(r"http\S+|www\S+", " ", str(text))

def remove_html(text: str) -> str:
    """
    Remove HTML tags and decode HTML entities.
    """

    text = html.unescape(text)
    return re.sub(r"<.*?>", " ", text)

def remove_punctuation(text: str) -> str:
    """
    Remove punctuation characters.
    """

    punctuation = f"[{re.escape(string.punctuation)}]"
    return re.sub(punctuation, " ", text)

def to_lowercase(text: str) -> str:
    """
    Convert text to lowercase.
    """

    return text.lower()

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple spaces with a single space.
    """

    return re.sub(r"\s+", " ", text).strip()

def clean_text(
    text: str,
    lowercase: bool = False,
    remove_punct: bool = False,
) -> str:
    """
    Clean a single text string.

    Parameters
    ----------
    text : str
        Input text.

    lowercase : bool
        Convert text to lowercase.

    remove_punct : bool
        Remove punctuation.

    Returns
    -------
    str
        Cleaned text.
    """

    text = str(text)
    text = remove_html(text)
    text = remove_urls(text)

    if lowercase:
        text = to_lowercase(text)

    if remove_punct:
        text = remove_punctuation(text)

    text = normalize_whitespace(text)

    return text

def preprocess_feedback(
    dataframe: pd.DataFrame,
    text_column: str = "text",
    lowercase: bool = False,
    remove_punct: bool = False,
) -> pd.DataFrame:
    """
    Preprocess the feedback dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    text_column : str
        Name of the text column.

    Returns
    -------
    pd.DataFrame
        Processed dataset.
    """

    dataframe = dataframe.copy()

    dataframe["clean_text"] = (
        dataframe[text_column]
        .fillna("")
        .astype(str)
        .apply(
            lambda text: clean_text(
                text,
                lowercase=lowercase,
                remove_punct=remove_punct,
            )
        )
    )

    dataframe = dataframe[
        dataframe["clean_text"].str.len() > 0
    ]

    return dataframe.reset_index(drop=True)