"""
model_utils.py

Utilities for saving and loading machine learning models.
"""

from pathlib import Path

import joblib

import numpy as np
import torch
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight

from transformers import Trainer

from bertopic import BERTopic
from gliner import GLiNER

from transformers import pipeline

def save_model(
    model,
    filepath: Path
):
    """
    Save a machine learning model.
    """

    joblib.dump(model, filepath)

def load_model(
    filepath: Path
):
    """
    Load a saved machine learning model.
    """

    return joblib.load(filepath)

def save_transformer_model(
    trainer,
    tokenizer,
    output_dir: Path
):
    """
    Save a HuggingFace model and tokenizer.
    """

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_sentiment_pipeline(
    model_directory: Path
):
    """
    Load a fine-tuned sentiment pipeline.
    """

    return pipeline(
        "text-classification",
        model=model_directory,
        tokenizer=model_directory
    )


def compute_class_weights(labels):
    """
    Compute balanced class weights for imbalanced datasets.

    Parameters
    ----------
    labels : array-like
        Encoded training labels.

    Returns
    -------
    torch.Tensor
        Class weights for CrossEntropyLoss.
    """

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    return torch.tensor(
        weights,
        dtype=torch.float
    )

class WeightedTrainer(Trainer):
    """
    Hugging Face Trainer with weighted cross-entropy loss.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def save_topic_model(
    topic_model,
    save_directory
):
    """
    Save BERTopic model.
    """

    topic_model.save(
        str(save_directory),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


def load_topic_model(
    save_directory
):
    """
    Load BERTopic model.
    """

    return BERTopic.load(
        str(save_directory)
    )


def save_gliner_model(
    model,
    save_directory
):
    """
    Save GLiNER model.

    NOTE:
    GLiNER models are normally downloaded
    from HuggingFace. Saving locally is optional.
    """

    model.save_pretrained(
        str(save_directory)
    )


def load_local_gliner_model(
    save_directory
):
    """
    Load locally saved GLiNER model.
    """

    return GLiNER.from_pretrained(
        str(save_directory)
    )