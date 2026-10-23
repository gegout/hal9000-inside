"""Text and spoken word normalization and counting utilities for HAL 9000 Inside."""

import re
import unicodedata
from typing import List, Optional, Tuple

# Regex to match spoken words: alphanumeric sequences, including hyphenated words and contractions
WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")

# Sentence splitting regex
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_spoken_text(text: str) -> str:
    """Normalize text intended for spoken delivery:
    - Strips markdown formatting (bold, italic, code blocks, bullet points, headers)
    - Strips brackets, URLs, excessive whitespace
    - Normalizes unicode characters
    """
    if not text:
        return ""

    # Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Strip code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)

    # Strip markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Strip markdown headers, list markers, blockquotes
    text = re.sub(r"^[#*>\-+\d\.]+\s+", "", text, flags=re.MULTILINE)

    # Strip markdown emphasis (*, **, _, __)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)

    # Strip system tags or debug prefixes e.g. [System State: ...]
    text = re.sub(r"\[.*?\]", "", text)

    # Collapse multiple whitespaces/newlines to single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_spoken_words(text: str) -> List[str]:
    """Extract spoken word tokens from text after normalization."""
    clean = normalize_spoken_text(text)
    return WORD_RE.findall(clean)


def count_spoken_words(text: str) -> int:
    """Return exact count of spoken words in the text."""
    return len(extract_spoken_words(text))


def truncate_to_word_limit(text: str, max_words: int = 20) -> str:
    """Enforce spoken word ceiling on a local response:
    - Retains complete sentences within the limit when possible.
    - Never speaks cut-off fragments.
    - If the first sentence alone exceeds max_words, returns a brief appropriate fallback
      or empty string rather than an incomplete fragment.
    """
    cleaned = normalize_spoken_text(text)
    if not cleaned:
        return ""

    total_words = count_spoken_words(cleaned)
    if total_words <= max_words:
        return cleaned

    # Split into sentences
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    retained_sentences = []
    current_word_count = 0

    for s in sentences:
        s_words = count_spoken_words(s)
        if current_word_count + s_words <= max_words:
            retained_sentences.append(s)
            current_word_count += s_words
        else:
            break

    if retained_sentences:
        return " ".join(retained_sentences)

    # If even the first sentence exceeds max_words, check if we can safely provide a brief complete sentence
    # or return empty to let the substantive response proceed
    return ""


def validate_local_response(text: str, max_words: int = 20, fallback: str = "") -> str:
    """Validate and enforce the hard word ceiling for local responses.
    Never returns cut-off fragments. If invalid or truncated to empty, returns fallback or empty.
    """
    res = truncate_to_word_limit(text, max_words=max_words)
    if not res and fallback:
        fb_words = count_spoken_words(fallback)
        if fb_words <= max_words:
            return fallback
    return res
