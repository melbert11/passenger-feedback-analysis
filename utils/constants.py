"""
constants.py

Contains reusable constants used throughout the NLP pipeline.
"""

# -----------------------------
# Sentiment Labels
# -----------------------------

SENTIMENT_LABELS = [
    "negative",
    "neutral",
    "positive"
]


# -----------------------------
# GLiNER Entity Labels
# -----------------------------

NER_LABELS = [
    "driver behavior",
    "driving quality",
    "safety issue",
    "vehicle condition",
    "cleanliness",
    "customer service",
    "communication",
    "punctuality",
    "traffic condition",
    "navigation issue",
    "payment issue",
    "fare issue",
    "pickup issue",
    "dropoff issue",
    "ride comfort"
]


# -----------------------------
# Topic Modeling
# -----------------------------

TOP_N_WORDS = 10
MIN_TOPIC_SIZE = 30
NR_TOPICS = None