"""Wikipedia integration for HAL 9000."""

from .client import (
    WikipediaClient,
    WikipediaArticle,
    WikipediaSearchResult,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
)
from .knowledge import (
    KnowledgeClient,
    ProviderError,
    SOURCES,
    SITES,
    record,
    plain,
    WIKIDATA_PROPERTY_LABELS,
)

__all__ = [
    "WikipediaClient",
    "WikipediaArticle",
    "WikipediaSearchResult",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "KnowledgeClient",
    "ProviderError",
    "SOURCES",
    "SITES",
    "record",
    "plain",
    "WIKIDATA_PROPERTY_LABELS",
]
