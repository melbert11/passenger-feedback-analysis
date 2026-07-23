"""
feature_engineering.py

Feature engineering utilities for textual feedback.
"""

import pandas as pd

def add_character_count(
    dataframe: pd.DataFrame,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Add the number of characters in each feedback.
    """

    dataframe = dataframe.copy()

    dataframe["character_count"] = (
        dataframe[text_column].str.len()
    )

    return dataframe

def add_word_count(
    dataframe: pd.DataFrame,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Add the number of words in each feedback.
    """

    dataframe = dataframe.copy()

    dataframe["word_count"] = (
        dataframe[text_column].str.split().str.len()
    )

    return dataframe

def add_average_word_length(
    dataframe: pd.DataFrame,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Compute the average word length.
    """

    dataframe = dataframe.copy()

    dataframe["average_word_length"] = (
        dataframe[text_column].apply(
            lambda text: (
                sum(len(word) for word in text.split()) / max(len(text.split()), 1)
            )
        )
    )

    return dataframe

def engineer_features(
    dataframe: pd.DataFrame,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    """

    dataframe = add_character_count(
        dataframe,
        text_column
    )

    dataframe = add_word_count(
        dataframe,
        text_column
    )

    dataframe = add_average_word_length(
        dataframe,
        text_column
    )

    return dataframe