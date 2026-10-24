"""Text and spoken word normalization, validation, and sanitization utilities for HAL 9000 Inside."""

import re
import unicodedata
from typing import List, Optional, Tuple

# Regex to match spoken words: alphanumeric sequences, including hyphenated words and contractions
WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")

# Sentence splitting regex
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Instruction leakage patterns
LEAKED_INSTRUCTION_PATTERNS = [
    r"you are hal 9000",
    r"as an ai",
    r"as a language model",
    r"write only the next",
    r"at most 20 words",
    r"spoken opening",
    r"conversational purpose",
    r"turn context",
    r"needs research",
    r"recent dialogue",
    r"im_start",
    r"im_end",
    r"<\|",
    r"\|>",
    r"role:\s*system",
    r"role:\s*user",
    r"role:\s*assistant",
    r"\[local reflex\]",
    r"local reflex",
    r"remote model",
    r"routing",
]

# Invented identity patterns
INVENTED_IDENTITY_PATTERNS = [
    r"time traveler",
    r"year 2256",
    r"2256",
    r"holographic ai",
    r"c\+\+ ai",
    r"cancun ai",
    r"al pacino",
    r"my name is [a-z]+ [a-z]+",  # unless HAL 9000
    r"space station called mirrors",
    r"virtual human",
    r"limited ai model",
    r"i've been diving myself",
    r"i have been diving myself",
]


def normalize_spoken_text(text: str) -> str:
    """Normalize text intended for spoken delivery:
    - Strips markdown formatting (bold, italic, code blocks, bullet points, headers)
    - Strips brackets, URLs, role tags, quotation marks around full sentences
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

    # Strip system tags or debug prefixes e.g. [System State: ...], [Local reflex]
    text = re.sub(r"\[.*?\]", "", text)

    # Strip role labels e.g. "Assistant:", "HAL 9000:", "HAL:"
    text = re.sub(r"^(?:assistant|hal\s*9000|hal|system|user):\s*", "", text, flags=re.IGNORECASE)

    # Collapse multiple whitespaces/newlines to single space
    text = re.sub(r"\s+", " ", text).strip()

    # Strip surrounding quotes if the model wrapped its output in quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    return text


def extract_spoken_words(text: str) -> List[str]:
    """Extract spoken word tokens from text after normalization."""
    clean = normalize_spoken_text(text)
    return WORD_RE.findall(clean)


def count_spoken_words(text: str) -> int:
    """Return exact count of spoken words in the text."""
    return len(extract_spoken_words(text))


def detect_instruction_leakage(text: str) -> bool:
    """Detect if output contains leaked system instructions or prompt scaffolding."""
    clean = text.lower()
    for pat in LEAKED_INSTRUCTION_PATTERNS:
        if re.search(pat, clean):
            return True
    return False


def detect_invented_identity(text: str) -> bool:
    """Detect if output invents an identity or false biography."""
    clean = text.lower()
    for pat in INVENTED_IDENTITY_PATTERNS:
        if re.search(pat, clean):
            if "hal 9000" not in clean:
                return True
    return False


def detect_input_echoing(output_text: str, user_utterance: Optional[str]) -> bool:
    """Detect if the output merely repeats the user's input verbatim or almost verbatim."""
    if not user_utterance or not output_text:
        return False
    u_clean = re.sub(r"[^\w\s]", "", user_utterance.lower()).strip()
    o_clean = re.sub(r"[^\w\s]", "", output_text.lower()).strip()
    if not u_clean or not o_clean:
        return False
    # Exact echo
    if o_clean == u_clean:
        return True
    # Echo with minor padding e.g. "I'm in Cancun for the week."
    if len(u_clean) > 8 and (o_clean.startswith(u_clean) or o_clean.endswith(u_clean)):
        return True
    return False


def truncate_to_word_limit(text: str, max_words: int = 20) -> str:
    """Enforce spoken word ceiling on a local response:
    - Retains complete sentences within the limit when possible.
    - Never speaks cut-off fragments.
    - If the first sentence alone exceeds max_words, returns empty string
      rather than an incomplete fragment.
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

    return ""


def validate_local_response(
    text: str,
    max_words: int = 20,
    fallback: str = "",
    user_utterance: Optional[str] = None,
) -> str:
    """Validate and enforce output safety and word limit for local openings:
    - Rejects instruction leakage, role labels, input echoing, and invented identities.
    - Enforces hard 20-word limit without incomplete fragments.
    - Returns validated text or fallback.
    """
    cleaned = normalize_spoken_text(text)
    if not cleaned:
        return fallback

    # Check instruction leakage
    if detect_instruction_leakage(cleaned):
        return fallback

    # Check invented identity
    if detect_invented_identity(cleaned):
        return fallback

    # Check input echoing
    if detect_input_echoing(cleaned, user_utterance):
        return fallback

    # Word limit truncation
    res = truncate_to_word_limit(cleaned, max_words=max_words)
    if not res and fallback:
        fb_words = count_spoken_words(fallback)
        if fb_words <= max_words:
            return fallback
    return res


def reconcile_continuation(opening: str, continuation: str) -> str:
    """Reconcile remote continuation with committed opening:
    - Strips duplicated greetings or acknowledgments already spoken in opening.
    - Prevents repeating what the opening already said.
    - If opening was a bridge like 'Let me compare those two options for you.',
      and continuation starts with 'Certainly, I will compare those two options for you: ...',
      strips the redundant opening phrase from the continuation.
    """
    if not opening or not continuation:
        return continuation.strip()

    op_norm = normalize_spoken_text(opening).strip()
    cont_norm = continuation.strip()

    # Common redundant leading phrases in LLM continuations
    lead_prefixes = [
        r"^(certainly|of course|understood|affirmative|sure)[\.,!\s]+",
        r"^(let me|i will|i am going to|allow me to) (compare|analyze|explain|check|look into|answer)[\w\s]+?[\.:,]\s*",
        r"^(here is|here are|regarding your question)[\w\s]*?[\.:,]\s*",
    ]

    for pat in lead_prefixes:
        cont_norm = re.sub(pat, "", cont_norm, flags=re.IGNORECASE).strip()

    # Capitalize first letter of continuation
    if cont_norm and cont_norm[0].islower():
        cont_norm = cont_norm[0].upper() + cont_norm[1:]

    return cont_norm
