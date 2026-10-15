"""Native asynchronous MediaWiki client for Wikipedia integration in HAL 9000.

Provides direct MediaWiki API calls using a single shared httpx.AsyncClient:
- search(query, limit): uses action=query, list=search, srnamespace=0.
- fetch(page_id): uses prop=extracts|info|pageprops, explaintext=1, inprop=url, ppprop=disambiguation.
- Returns structured results with title, page ID, actual URL, revision ID, language,
  retrieval timestamp, bounded text, and truncation status.
- English ('en') default with validated French ('fr') language support.
- Explicit handling of missing articles and disambiguation pages.
- Descriptive User-Agent identification for HAL 9000.
- Timeouts, bounded retries respecting Retry-After, cancellation, concurrency limits.
- SQLite caching with TTL expiry, explicit distinction of empty results vs network/API failures,
  and labeling of stale cached results.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "fr"}
DEFAULT_LANGUAGE = "en"
DEFAULT_USER_AGENT = (
    "HAL9000-Inside/0.3.1 (https://github.com/cgegout/hal9000-inside; contact@hal9000.space)"
)
DEFAULT_MAX_TEXT_CHARS = 4000
DEFAULT_CACHE_TTL_SECONDS = 86400 * 7  # 7 days
DEFAULT_TIMEOUT_SECONDS = 12.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_CONCURRENCY = 4


@dataclass
class WikipediaSearchResult:
    """A search candidate returned by Wikipedia search."""
    title: str
    page_id: int
    snippet: str
    word_count: int
    language: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WikipediaArticle:
    """Structured article extracted from Wikipedia."""
    title: str
    page_id: int
    url: str
    revision_id: Optional[int]
    language: str
    retrieval_timestamp: str
    extract: str
    truncated: bool
    is_disambiguation: bool = False
    is_missing: bool = False
    is_stale: bool = False
    status: str = "ok"  # 'ok', 'missing', 'disambiguation', 'error'
    error_message: Optional[str] = None
    disambiguation_options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WikipediaClient:
    """Shared asynchronous Wikipedia client for HAL 9000."""

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        concurrency: int = DEFAULT_CONCURRENCY,
        cache_db_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.language = self._validate_language(language)
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_text_chars = max_text_chars
        self.cache_ttl_seconds = cache_ttl_seconds

        self._sem = asyncio.Semaphore(concurrency)
        self._external_client = client is not None
        self._client: Optional[httpx.AsyncClient] = client

        # Cache DB location
        if cache_db_path:
            self.cache_db_path = cache_db_path
        else:
            base_dir = data_dir or os.environ.get("SNAP_USER_COMMON") or os.path.expanduser("~/.local/share/hal9000-inside")
            self.cache_db_path = os.path.join(base_dir, "wikipedia_cache.db")

        self._init_cache_db()

    @staticmethod
    def _validate_language(lang: str) -> str:
        clean = (lang or "").strip().lower()
        if clean not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Unsupported Wikipedia language '{lang}'. Supported: {sorted(SUPPORTED_LANGUAGES)}. Defaulting to '{DEFAULT_LANGUAGE}'."
            )
            return DEFAULT_LANGUAGE
        return clean

    def _init_cache_db(self):
        """Create SQLite cache table if not exists."""
        try:
            parent = os.path.dirname(self.cache_db_path)
            if parent:
                os.makedirs(parent, exist_ok=True, mode=0o700)
            conn = sqlite3.connect(self.cache_db_path, timeout=10.0)
            with conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS wiki_cache (
                        key TEXT PRIMARY KEY,
                        cache_type TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_wiki_cache_type ON wiki_cache (cache_type);")
            conn.close()
        except Exception as e:
            logger.warning(f"Could not initialize Wikipedia SQLite cache at {self.cache_db_path}: {e}")

    def _get_from_cache(self, key: str) -> Optional[Tuple[Dict[str, Any], bool]]:
        """Retrieve item from cache. Returns (data_dict, is_stale) or None."""
        try:
            conn = sqlite3.connect(self.cache_db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data_json, expires_at FROM wiki_cache WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            data_json, expires_at = row
            now = time.time()
            is_stale = now > expires_at
            parsed = json.loads(data_json)
            return parsed, is_stale
        except Exception as e:
            logger.debug(f"Cache read error for key {key}: {e}")
            return None

    def _save_to_cache(self, key: str, cache_type: str, data: Dict[str, Any], ttl_seconds: Optional[int] = None):
        """Store item in cache with TTL."""
        try:
            ttl = ttl_seconds if ttl_seconds is not None else self.cache_ttl_seconds
            now = time.time()
            expires_at = now + ttl
            data_json = json.dumps(data)
            conn = sqlite3.connect(self.cache_db_path, timeout=5.0)
            with conn:
                conn.execute(
                    """
                    INSERT INTO wiki_cache (key, cache_type, data_json, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        data_json = excluded.data_json,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at
                    """,
                    (key, cache_type, data_json, now, expires_at),
                )
            conn.close()
        except Exception as e:
            logger.debug(f"Cache write error for key {key}: {e}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or initialize the shared httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                follow_redirects=True,
            )
        return self._client

    def _get_api_endpoint(self, lang: Optional[str] = None) -> str:
        eff_lang = self._validate_language(lang or self.language)
        return f"https://{eff_lang}.wikipedia.org/w/api.php"

    async def _request_with_retry(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> httpx.Response:
        """Send HTTP GET with concurrency limiting, timeouts, and bounded retries honoring Retry-After."""
        retries = max_retries if max_retries is not None else self.max_retries
        client = await self._get_client()

        for attempt in range(retries + 1):
            async with self._sem:
                try:
                    resp = await client.get(endpoint, params=params)
                except (httpx.TimeoutException, httpx.NetworkError) as ne:
                    if attempt < retries:
                        backoff = (2 ** attempt) * 0.5
                        logger.warning(
                            f"Wikipedia request to {endpoint} failed ({ne}), retrying in {backoff:.1f}s (attempt {attempt+1}/{retries})..."
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise

                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < retries:
                        retry_after_hdr = resp.headers.get("Retry-After")
                        try:
                            delay = float(retry_after_hdr) if retry_after_hdr else ((2 ** attempt) * 0.5)
                        except ValueError:
                            delay = (2 ** attempt) * 0.5
                        # Cap backoff to reasonable maximum
                        delay = min(delay, 10.0)
                        logger.warning(
                            f"Wikipedia HTTP {resp.status_code} on {endpoint}, waiting {delay:.1f}s before retry..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                return resp

        raise RuntimeError(f"Exhausted {retries} retries connecting to Wikipedia.")

    async def search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[WikipediaSearchResult]:
        """Search Wikipedia using action=query, list=search, srnamespace=0.
        
        Returns a list of structured WikipediaSearchResult objects.
        """
        clean_q = (query or "").strip()
        if not clean_q:
            return []

        eff_lang = self._validate_language(language or self.language)
        cache_key = f"search:{eff_lang}:{limit}:{clean_q.lower()}"

        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                cached_data, is_stale = cached
                if not is_stale:
                    return [WikipediaSearchResult(**item) for item in cached_data]

        endpoint = self._get_api_endpoint(eff_lang)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_q,
            "srlimit": max(1, min(limit, 20)),
            "srnamespace": 0,
            "format": "json",
            "utf8": 1,
        }

        try:
            resp = await self._request_with_retry(endpoint, params)
            data = resp.json()
        except Exception as e:
            logger.error(f"Wikipedia search failed for '{clean_q}': {e}")
            # If stale cached result exists, return it on failure as fallback
            cached = self._get_from_cache(cache_key)
            if cached:
                cached_data, _ = cached
                logger.info(f"Returning stale cached search results for '{clean_q}' after network error")
                return [WikipediaSearchResult(**item) for item in cached_data]
            raise

        search_items = data.get("query", {}).get("search", [])
        results: List[WikipediaSearchResult] = []
        for item in search_items:
            results.append(
                WikipediaSearchResult(
                    title=item.get("title", ""),
                    page_id=int(item.get("pageid", 0)),
                    snippet=item.get("snippet", ""),
                    word_count=int(item.get("wordcount", 0)),
                    language=eff_lang,
                )
            )

        if use_cache:
            self._save_to_cache(
                cache_key,
                "search",
                [r.to_dict() for r in results],
            )

        return results

    async def fetch(
        self,
        page_id: int,
        language: Optional[str] = None,
        max_chars: Optional[int] = None,
        use_cache: bool = True,
    ) -> WikipediaArticle:
        """Fetch article by page ID using action=query, prop=extracts|info|pageprops.
        
        Returns structured WikipediaArticle with explicit missing and disambiguation detection.
        """
        eff_lang = self._validate_language(language or self.language)
        eff_max_chars = max_chars or self.max_text_chars
        cache_key = f"fetch:{eff_lang}:{page_id}"
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                cached_data, is_stale = cached
                if not is_stale:
                    art = WikipediaArticle(**cached_data)
                    art.is_stale = False
                    return art

        endpoint = self._get_api_endpoint(eff_lang)
        params = {
            "action": "query",
            "prop": "extracts|info|pageprops",
            "explaintext": 1,
            "inprop": "url",
            "ppprop": "disambiguation",
            "pageids": page_id,
            "format": "json",
            "utf8": 1,
        }

        try:
            resp = await self._request_with_retry(endpoint, params)
            data = resp.json()
        except Exception as e:
            logger.error(f"Wikipedia fetch failed for page_id {page_id}: {e}")
            # Check for stale cache fallback
            cached = self._get_from_cache(cache_key)
            if cached:
                cached_data, _ = cached
                logger.info(f"Returning stale cached article for page_id {page_id} after network failure")
                art = WikipediaArticle(**cached_data)
                art.is_stale = True
                return art
            return WikipediaArticle(
                title=f"Page {page_id}",
                page_id=page_id,
                url=f"https://{eff_lang}.wikipedia.org/?curid={page_id}",
                revision_id=None,
                language=eff_lang,
                retrieval_timestamp=now_ts,
                extract="",
                truncated=False,
                status="error",
                error_message=f"Network or API failure: {e}",
            )

        pages = data.get("query", {}).get("pages", {})
        page_data = pages.get(str(page_id)) or {}

        # 1. Missing page check
        if not page_data or "missing" in page_data:
            art = WikipediaArticle(
                title=page_data.get("title") or f"Page {page_id}",
                page_id=page_id,
                url=f"https://{eff_lang}.wikipedia.org/?curid={page_id}",
                revision_id=None,
                language=eff_lang,
                retrieval_timestamp=now_ts,
                extract="",
                truncated=False,
                is_missing=True,
                status="missing",
                error_message="The requested Wikipedia article does not exist or has been removed.",
            )
            if use_cache:
                self._save_to_cache(cache_key, "fetch", art.to_dict(), ttl_seconds=3600)  # Shorter TTL for missing
            return art

        title = page_data.get("title", f"Page {page_id}")
        full_url = page_data.get("fullurl") or f"https://{eff_lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        last_revid = page_data.get("lastrevid")
        raw_extract = page_data.get("extract", "") or ""

        # 2. Disambiguation check
        is_disambig = "disambiguation" in page_data.get("pageprops", {})
        disambig_options: List[str] = []
        if is_disambig:
            # Parse possible choices from lines
            lines = [line.strip() for line in raw_extract.split("\n") if line.strip()]
            disambig_options = lines[:15]

        # 3. Text bounding and truncation
        truncated = False
        if len(raw_extract) > eff_max_chars:
            extract = raw_extract[:eff_max_chars]
            truncated = True
        else:
            extract = raw_extract

        status = "disambiguation" if is_disambig else "ok"

        art = WikipediaArticle(
            title=title,
            page_id=page_id,
            url=full_url,
            revision_id=last_revid,
            language=eff_lang,
            retrieval_timestamp=now_ts,
            extract=extract,
            truncated=truncated,
            is_disambiguation=is_disambig,
            is_missing=False,
            is_stale=False,
            status=status,
            disambiguation_options=disambig_options,
        )

        if use_cache:
            self._save_to_cache(cache_key, "fetch", art.to_dict())

        return art

    async def search_and_fetch_best(
        self,
        query: str,
        language: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> Optional[WikipediaArticle]:
        """Search Wikipedia, filter out disambiguation matches where possible, and fetch the best candidate."""
        results = await self.search(query, limit=5, language=language)
        if not results:
            return None

        # Filter out titles explicitly marked as disambiguation
        candidates = [r for r in results if "(disambiguation)" not in r.title.lower()]
        target = candidates[0] if candidates else results[0]

        return await self.fetch(target.page_id, language=language, max_chars=max_chars)

    async def close(self):
        """Close the internal HTTP client."""
        if self._client is not None and not self._client.is_closed and not self._external_client:
            await self._client.aclose()
            self._client = None
