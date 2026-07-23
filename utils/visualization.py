"""
visualization.py

Reusable visualization functions for exploratory data analysis,
natural language processing, and model evaluation.
"""

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from sklearn.metrics import ConfusionMatrixDisplay

def plot_sentiment_distribution(
    dataframe: pd.DataFrame,
    column: str = "sentiment",
):
    """
    Plot sentiment distribution.
    """

    figure, axis = plt.subplots(figsize=(7, 5))

    dataframe[column].value_counts().plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title("Sentiment Distribution")
    axis.set_xlabel("Sentiment")
    axis.set_ylabel("Count")

    figure.tight_layout()

    return figure

def plot_feedback_length(
    dataframe: pd.DataFrame,
    column: str = "character_count",
):
    """
    Plot distribution of feedback length.
    """

    figure, axis = plt.subplots(figsize=(7,5))

    dataframe[column].plot.hist(
        bins=30,
        ax=axis,
    )

    axis.set_title("Character Count Distribution")
    axis.set_xlabel("Characters")

    figure.tight_layout()

    return figure

def plot_word_count(
    dataframe: pd.DataFrame,
    column: str = "word_count",
):
    """
    Plot word count distribution.
    """

    figure, axis = plt.subplots(figsize=(7,5))

    dataframe[column].plot.hist(
        bins=30,
        ax=axis,
    )

    axis.set_title("Word Count Distribution")

    figure.tight_layout()

    return figure

def plot_top_words(
    dataframe: pd.DataFrame,
    text_column: str = "clean_text",
    top_n: int = 20,
):
    """
    Plot the most frequent words.
    """

    words = " ".join(dataframe[text_column]).split()
    frequencies = Counter(words)
    top_words = frequencies.most_common(top_n)

    labels = [word for word, _ in top_words]
    values = [count for _, count in top_words]

    figure, axis = plt.subplots(figsize=(10,6))

    axis.bar(labels, values)
    axis.set_title(f"Top {top_n} Words")
    axis.tick_params(axis="x", rotation=45)

    figure.tight_layout()

    return figure

def plot_wordcloud(
    dataframe: pd.DataFrame,
    text_column: str = "clean_text",
):
    """
    Generate a word cloud.
    """

    text = " ".join(dataframe[text_column])

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
    ).generate(text)

    figure, axis = plt.subplots(figsize=(10,5))

    axis.imshow(wordcloud)
    axis.axis("off")
    figure.tight_layout()

    return figure

def plot_confusion_matrix(
    confusion_matrix,
    labels,
):
    """
    Display a confusion matrix.
    """

    figure, axis = plt.subplots(figsize=(6,6))

    ConfusionMatrixDisplay(
        confusion_matrix,
        display_labels=labels,
    ).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )

    axis.set_title("Confusion Matrix")
    figure.tight_layout()

    return figure

def plot_entity_distribution(entity_df):
    """
    Plot entity frequency.
    """

    counts = entity_df["label"].value_counts()

    fig, ax = plt.subplots(figsize=(10,5))

    counts.plot.bar(ax=ax)

    ax.set_title("Entity Distribution")
    ax.set_xlabel("Entity")
    ax.set_ylabel("Count")

    plt.tight_layout()

    return fig


def plot_top_entities(
    entity_df,
    top_n=10
):
    """
    Plot most frequent extracted entities.
    """

    counts = (
        entity_df["entity"]
        .value_counts()
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    counts.plot.bar(ax=ax)
    ax.set_title("Top Extracted Entities")
    plt.tight_layout()

    return fig


def plot_topic_distribution(topic_info):
    """
    Plot BERTopic topic sizes.
    """

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        topic_info["Topic"].astype(str),
        topic_info["Count"]
    )

    ax.set_title("Topic Distribution")
    plt.tight_layout()

    return fig