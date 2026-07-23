"""
ner_utils.py

Utility functions for Named Entity Recognition using GLiNER.
"""

import pandas as pd
from gliner import GLiNER


def load_gliner_model(
    model_name="urchade/gliner_medium-v2.1"
):
    """
    Load the GLiNER model.

    Parameters
    ----------
    model_name : str
        HuggingFace GLiNER model.

    Returns
    -------
    GLiNER
    """
    return GLiNER.from_pretrained(model_name)


def extract_entities(
    text,
    model,
    labels,
    threshold=0.5
):
    """
    Extract entities from a single text.
    """

    return model.predict_entities(
        text,
        labels,
        threshold=threshold
    )


def extract_entities_dataframe(
    dataframe,
    text_column,
    model,
    labels,
    threshold=0.5
):
    """
    Extract entities for every row in a dataframe.
    """

    df = dataframe.copy()

    df["entities"] = df[text_column].apply(
        lambda x: extract_entities(
            x,
            model,
            labels,
            threshold
        )
    )

    return df


def flatten_entities(df):
    """
    Convert nested entity lists into a flat dataframe.
    """

    rows = []

    for _, row in df.iterrows():
        for entity in row["entities"]:
            rows.append({
                "feedback": row["feedback_text"],
                "entity": entity["text"],
                "label": entity["label"],
                "score": entity["score"]
            })

    return pd.DataFrame(rows)