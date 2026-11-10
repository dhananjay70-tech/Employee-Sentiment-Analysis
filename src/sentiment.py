"""Schema discovery, validation, and Hugging Face sentiment labeling."""
from __future__ import annotations

import logging
from typing import Iterable
import pandas as pd

logger = logging.getLogger(__name__)
TEXT_HINTS = ("message", "body", "text", "content", "email", "subject")
EMPLOYEE_HINTS = ("employee", "sender", "author", "person", "user", "name", "from")
DATE_HINTS = ("date", "time", "sent", "created", "timestamp")


def inspect_dataframe(df: pd.DataFrame) -> dict:
    """Return a compact, data-driven schema and quality profile."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names; rename them before analysis.")
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "candidates": {
            "text": suggest_columns(df.columns, TEXT_HINTS),
            "employee": suggest_columns(df.columns, EMPLOYEE_HINTS),
            "date": suggest_columns(df.columns, DATE_HINTS),
        },
    }


def resolve_column(df: pd.DataFrame, explicit: str | None, kind: str) -> str:
    """Resolve an explicitly selected column or a single unambiguous hint match."""
    hints = {"text": TEXT_HINTS, "employee": EMPLOYEE_HINTS, "date": DATE_HINTS}
    if kind not in hints:
        raise ValueError(f"Unknown column kind {kind!r}; expected one of {sorted(hints)}")
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"Configured {kind} column {explicit!r} is missing. Available columns: {list(df.columns)}")
        return explicit
    candidates = suggest_columns(df.columns, hints[kind])
    if len(candidates) != 1:
        raise ValueError(f"Could not resolve {kind} column unambiguously. Candidates: {candidates}; set it explicitly.")
    logger.info("Resolved %s column from its unique header match: %s", kind, candidates[0])
    return candidates[0]


def suggest_columns(columns: Iterable, hints: tuple[str, ...]) -> list[str]:
    return [str(c) for c in columns if any(h in str(c).lower() for h in hints)]


def classify_messages(texts: pd.Series, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", batch_size: int = 16) -> pd.Series:
    """Classify text as Positive/Negative/Neutral; empty text remains missing.

    Requires transformers and a model download on first use. Uses a public,
    three-class pretrained model; predictions are weak labels, not ground truth.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install requirements.txt to use Hugging Face sentiment inference.") from exc
    results = pd.Series(pd.NA, index=texts.index, dtype="string")
    valid = texts.notna() & texts.astype("string").str.strip().ne("")
    if not valid.any():
        return results
    logger.info("Loading sentiment model %s", model_name)
    try:
        classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
        values = texts.loc[valid].astype(str).tolist()
        predictions = classifier(values, batch_size=batch_size, truncation=True, max_length=512)
    except Exception:
        logger.exception("Sentiment inference failed for model %s", model_name)
        raise
    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    mapped = []
    for prediction in predictions:
        key = str(prediction["label"]).lower()
        if key not in label_map:
            raise ValueError(f"Unexpected model label {prediction['label']!r}; expected three sentiment classes.")
        mapped.append(label_map[key])
    results.loc[valid] = mapped
    logger.info("Labeled %d messages; left %d blank messages unlabeled", int(valid.sum()), int((~valid).sum()))
    return results
