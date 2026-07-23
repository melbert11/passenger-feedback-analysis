# Passenger Feedback Analysis

An NLP notebook project for analyzing passenger feedback. The workflow cleans raw
ride-feedback records, fine-tunes a sentiment classifier, extracts domain-specific
entities, and identifies recurring topics.

## Workflow

Run the notebooks in order because each stage writes inputs used by the next one.

| Step | Notebook | Purpose | Main output |
| --- | --- | --- | --- |
| 1 | `01_data_preparation.ipynb` | Cleans feedback text and adds text-length features. | `data/processed/feedback_clean.csv` |
| 2 | `02_sentiment_analysis.ipynb` | Fine-tunes a RoBERTa sentiment classifier and evaluates it. | Saved sentiment model and label encoder in `artifacts/` |
| 3 | `03_named_entity_extraction.ipynb` | Uses GLiNER to identify feedback entities such as safety, cleanliness, and fare issues. | Processed entity datasets |
| 4 | `04_topic_modeling.ipynb` | Uses BERTopic to discover themes across feedback. | Topic datasets and a BERTopic model |

## Repository layout

```text
data/
  raw/feedback.csv              # versioned source dataset
  processed/                    # generated cleaned/enriched datasets
notebooks/                      # analysis workflow, in execution order
utils/                          # reusable loading, preprocessing, model, and plotting helpers
artifacts/                      # generated encoders, transformer checkpoints, vectorizers, and scalers
models/                         # generated topic-model files
outputs/                        # generated figures and reports
```

`utils/config.py` defines the project paths and creates the generated output
directories when it is imported.

## Data

The input dataset is stored at `data/raw/feedback.csv` and is intentionally kept
in version control. Its records include `trip_id`, `timestamp`, `rating`,
`feedback_text`, and `sentiment`. The preparation notebook turns `feedback_text`
into cleaned text and creates character-count, word-count, and average-word-length
features.

Do not commit derived data or model outputs. `.gitignore` excludes
`data/processed/`, `artifacts/`, and `models/` because they can be regenerated and
may be large.

## Setup

This project targets Python 3.13 and uses `uv` for dependency management.

```bash
uv sync
uv run jupyter notebook
```

Open the notebooks from the repository root so that imports such as
`from utils.data_loader import load_processed_feedback` resolve correctly.

The sentiment, entity-extraction, and topic-modeling notebooks also rely on their
respective ML libraries and model downloads (for example PyTorch, GLiNER,
BERTopic, and sentence-transformers). Install any missing notebook dependencies in
the environment where you run them.

## Sentiment model and Google Colab

Sentiment training uses
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
It was run in Google Colab because downloading and saving the fine-tuned model is
storage-intensive. The repository deliberately does not include the resulting
model weights or tokenizer files.

To reproduce the sentiment stage, open `02_sentiment_analysis.ipynb` in Colab (or
another environment with sufficient storage and compute), install the required
packages, and run it after the preparation notebook. It writes the model to
`artifacts/transformers/sentiment_model/` and the label encoder to
`artifacts/encoders/sentiment_label_encoder.pkl`; both paths are ignored by Git.

