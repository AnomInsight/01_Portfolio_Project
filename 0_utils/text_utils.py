"""
text_utils.py
-------------
General-purpose text preprocessing utilities for NLP projects.

Use cases: sentiment analysis, text classification, document similarity,
spam filtering, intent detection.

Typical usage
-------------
from text_utils import clean_text, tokenize, pad_sequences_np, build_vocab

texts_clean = [clean_text(t) for t in raw_texts]
vocab = build_vocab(texts_clean, max_vocab=10000)
sequences = [text_to_sequence(t, vocab) for t in texts_clean]
X = pad_sequences_np(sequences, max_len=100)
"""

import re
import string
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_digits: bool = False,
    remove_urls: bool = True,
    remove_extra_whitespace: bool = True,
) -> str:
    """Basic text cleaning pipeline."""
    if lowercase:
        text = text.lower()
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if remove_digits:
        text = re.sub(r"\d+", " ", text)
    if remove_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str, stopwords: set[str]) -> str:
    """Remove stopwords from a text string."""
    return " ".join(w for w in text.split() if w not in stopwords)


# ---------------------------------------------------------------------------
# Tokenization and vocabulary
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Split text into word tokens (whitespace)."""
    return text.split()


def build_vocab(
    texts: list[str],
    max_vocab: int = 10_000,
    min_freq: int = 1,
    special_tokens: list[str] | None = None,
) -> dict[str, int]:
    """Build a word-to-index vocabulary from a list of texts.

    Returns
    -------
    vocab dict: word -> int index (0=PAD, 1=UNK by default)
    """
    special_tokens = special_tokens or ["<PAD>", "<UNK>"]
    counter = Counter(word for text in texts for word in tokenize(text))
    most_common = [w for w, c in counter.most_common(max_vocab) if c >= min_freq]
    vocab = {tok: i for i, tok in enumerate(special_tokens)}
    for word in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text: str, vocab: dict[str, int], unk_idx: int = 1) -> list[int]:
    """Convert a text string to a list of integer indices."""
    return [vocab.get(word, unk_idx) for word in tokenize(text)]


# ---------------------------------------------------------------------------
# Padding and truncation
# ---------------------------------------------------------------------------

def pad_sequences_np(
    sequences: list[list[int]],
    max_len: int,
    pad_value: int = 0,
    truncate: str = "post",
    padding: str = "post",
) -> np.ndarray:
    """Pad/truncate a list of sequences to a fixed length.

    Parameters
    ----------
    truncate : 'pre' or 'post' — where to cut long sequences
    padding  : 'pre' or 'post' — where to add padding tokens
    """
    result = np.full((len(sequences), max_len), pad_value, dtype=np.int32)

    for i, seq in enumerate(sequences):
        if truncate == "post":
            seq = seq[:max_len]
        else:
            seq = seq[-max_len:]

        if padding == "post":
            result[i, :len(seq)] = seq
        else:
            result[i, max_len - len(seq):] = seq

    return result


# ---------------------------------------------------------------------------
# Bag of Words / TF-IDF (sklearn wrappers)
# ---------------------------------------------------------------------------

def fit_tfidf(texts: list[str], max_features: int = 10_000, **kwargs):
    """Fit and return a TF-IDF vectorizer and transformed matrix.

    Returns
    -------
    vectorizer, X_tfidf (sparse matrix)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_features, **kwargs)
    X = vec.fit_transform(texts)
    return vec, X


def transform_tfidf(vectorizer, texts: list[str]):
    """Apply a fitted TF-IDF vectorizer to new texts."""
    return vectorizer.transform(texts)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def text_length_summary(texts: list[str]) -> dict:
    """Return min/max/mean/median word-count stats across texts."""
    lengths = [len(tokenize(t)) for t in texts]
    return {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
    }
