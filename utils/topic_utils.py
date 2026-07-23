"""
topic_utils.py

Utility functions for BERTopic.
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def load_embedding_model(
    model_name="all-MiniLM-L6-v2"
):
    """
    Load Sentence Transformer model.
    """

    return SentenceTransformer(model_name)


def train_topic_model(
    documents,
    embedding_model,
    min_topic_size=10,
    nr_topics=None
):
    documents = documents.tolist() if hasattr(documents, "tolist") else list(documents)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics, # Change from "auto" to an int or None
        calculate_probabilities=True
    )

    topics, probabilities = topic_model.fit_transform(documents)

    topic_info = topic_model.get_topic_info()

    return topic_model, topics, probabilities, topic_info


def assign_topics(
    topic_model,
    documents
):
    """
    Assign topics to documents.
    """

    return topic_model.transform(documents)


def get_topic_info(
    topic_model
):
    """
    Return topic summary dataframe.
    """

    return topic_model.get_topic_info()