"""HAL 4-adapter KnowledgeClient integrating Wikipedia, Wikidata, arXiv, and Stack Exchange.

Pure native async client using a single shared httpx.AsyncClient.
No browser, Node.js, external MCP subprocess, or mandatory API key.
Source material is untrusted evidence, never verified facts.
"""

from __future__ import annotations

import asyncio
import copy
import html
import json
import logging
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)

SOURCES = ("wikipedia", "wikidata", "arxiv", "stackexchange")
SITES = ("stackoverflow", "askubuntu", "unix", "superuser", "serverfault")
ATOM = {"a": "http://www.w3.org/2005/Atom"}
SUPPORTED_LANGUAGES = {"en", "fr"}
DEFAULT_LANGUAGE = "en"
DEFAULT_USER_AGENT = (
    "HAL9000-Inside/0.3.1 (https://github.com/cgegout/hal9000-inside; contact@hal9000.space)"
)

# Common Wikidata property labels for human-readable claims
WIKIDATA_PROPERTY_LABELS: Dict[str, str] = {
    "P31": "instance of",
    "P279": "subclass of",
    "P17": "country",
    "P131": "located in administrative entity",
    "P569": "date of birth",
    "P570": "date of death",
    "P571": "inception / founded",
    "P576": "dissolved / abolished",
    "P1082": "population",
    "P856": "official website",
    "P159": "headquarters location",
    "P112": "founded by",
    "P169": "chief executive officer",
    "P361": "part of",
    "P527": "has part",
    "P39": "position held",
    "P106": "occupation",
    "P138": "named after",
    "P61": "discoverer / inventor",
    "P575": "time of discovery / invention",
    "P625": "coordinate location",
    "P27": "country of citizenship",
    "P19": "place of birth",
    "P20": "place of death",
    "P18": "image",
    "P345": "IMDb ID",
    "P214": "VIAF ID",
    "P244": "LCAuth ID",
    "P227": "GND ID",
    "P213": "ISNI",
}


class ProviderError(Exception):
    """Raised when an external knowledge provider fails or is rate-limited."""

    def __init__(self, message: str, status: str = "unavailable"):
        super().__init__(message)
        self.status = status


def plain(value: Optional[str]) -> str:
    """Strip HTML tags and unescape entities."""
    return html.unescape(re.sub(r"<[^>]+>", " ", value or "")).strip()


def record(
    source: str,
    identifier: Union[str, int],
    title: str,
    url: str,
    content: str = "",
    **metadata: Any,
) -> Dict[str, Any]:
    """Create a standardized evidence record."""
    return dict(
        source=source,
        identifier=str(identifier),
        title=title,
        url=url,
        content=content,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        verified=False,
        **metadata,
    )


class KnowledgeClient:
    """Four-adapter asynchronous Knowledge Client for HAL 9000 Inside.
    
    Supports:
    - Wikipedia (MediaWiki API: search, fetch page extract & disambiguation)
    - Wikidata (wbsearchentities, wbgetentities with statements, qualifiers, labels)
    - arXiv (Atom API: query, abstract, authors, date)
    - Stack Exchange (Stack API 2.3: search, question, accepted-answer / top-voted answers)
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: float = 15.0,
        retries: int = 2,
        ttl: float = 3600.0 * 24 * 7,  # 7 days
        stale_ttl: float = 86400.0 * 30,  # 30 days
        cooldown: float = 60.0,
        arxiv_spacing: float = 3.1,
        user_agent: str = DEFAULT_USER_AGENT,
        max_cache_entries: int = 10000,
        max_serialized_output_chars: int = 16000,
        max_output_chars: Optional[int] = None,
        default_language: str = DEFAULT_LANGUAGE,
    ):
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.ttl = float(ttl)
        self.stale_ttl = float(stale_ttl)
        self.cooldown = float(cooldown)
        self.spacing = float(arxiv_spacing)
        self.max_cache_entries = int(max_cache_entries)
        self.max_serialized_output_chars = int(max_output_chars if max_output_chars is not None else max_serialized_output_chars)
        self.default_language = self._validate_language(default_language)
        self.user_agent = user_agent

        # HTTP Client
        self.http = httpx.AsyncClient(
            transport=transport,
            timeout=self.timeout,
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
        )

        # Cache DB path
        if cache_path:
            self.cache_path = cache_path
        else:
            base = (
                data_dir
                or os.environ.get("SNAP_USER_COMMON")
                or os.path.expanduser("~/.local/share/hal9000-inside")
            )
            self.cache_path = os.path.join(base, "knowledge_cache.sqlite3")

        self.db: Optional[sqlite3.Connection] = None
        self._init_db()

        # Concurrency & Rate Limiting
        self.locks = {s: asyncio.Lock() for s in SOURCES}
        self.until: Dict[str, float] = {}
        self.next_request: Dict[str, float] = {}
        self._inflight: Dict[str, asyncio.Future] = {}
        self._inflight_lock = asyncio.Lock()
        self.closed = False

    @staticmethod
    def _validate_language(lang: str) -> str:
        clean = (lang or "").strip().lower()
        if clean not in SUPPORTED_LANGUAGES:
            return DEFAULT_LANGUAGE
        return clean

    def _init_db(self):
        """Initialize SQLite database with WAL and index."""
        if self.cache_path != ":memory:":
            parent = os.path.dirname(self.cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.cache_path, timeout=10.0)
        with self.db:
            if self.cache_path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL;")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS knowledge_cache (key TEXT PRIMARY KEY, saved REAL, value TEXT)"
            )
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_kc_saved ON knowledge_cache (saved)")

    def _evict_cache_if_needed(self, max_entries: Optional[int] = None):
        """Size/age eviction: delete records older than stale_ttl, or oldest if size exceeded."""
        if not self.db or self.cache_path == ":memory:":
            return
        limit_entries = max_entries if max_entries is not None else self.max_cache_entries
        try:
            cutoff = time.time() - self.stale_ttl
            with self.db:
                self.db.execute("DELETE FROM knowledge_cache WHERE saved < ?", (cutoff,))
                count_row = self.db.execute("SELECT COUNT(*) FROM knowledge_cache").fetchone()
                total = count_row[0] if count_row else 0
                if total > limit_entries:
                    excess = total - limit_entries
                    self.db.execute(
                        "DELETE FROM knowledge_cache WHERE key IN (SELECT key FROM knowledge_cache ORDER BY saved ASC LIMIT ?)",
                        (excess,),
                    )
        except Exception as e:
            logger.debug(f"Cache eviction error: {e}")

    async def close(self):
        """Gracefully close HTTP client and SQLite connection."""
        if not self.closed:
            self.closed = True
            await self.http.aclose()
            if self.db:
                self.db.close()
                self.db = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    @staticmethod
    def route(query: str) -> str:
        """Route query to the best-suited provider heuristic."""
        q = query.lower()
        if any(w in q for w in ("traceback", "linux", "ubuntu", "python error", "stackoverflow", "exception", "apt", "bash")):
            return "stackexchange"
        if any(w in q for w in ("paper", "research", "preprint", "arxiv", "theorem", "arxiv:")):
            return "arxiv"
        if any(w in q for w in ("wikidata", "date of birth", "population", "founded", "q-number", "qid")):
            return "wikidata"
        return "wikipedia"

    async def search(
        self,
        query: str,
        source: str = "auto",
        limit: int = 5,
        language: Optional[str] = None,
        site: str = "stackoverflow",
        overall_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Search across knowledge providers."""
        if not query or not query.strip() or len(query) > 500:
            raise ValueError("Invalid query")
        eff_lang = self._validate_language(language or self.default_language)
        eff_source = self.route(query) if source == "auto" else source
        clamped_limit = max(1, min(int(limit), 10))

        coro = self._dedup_run(
            eff_source, "search", query.strip(), clamped_limit, eff_lang, site
        )
        if overall_timeout:
            raw_result = await asyncio.wait_for(coro, timeout=overall_timeout)
        else:
            raw_result = await coro

        result = copy.deepcopy(raw_result)
        self._bound_serialized_results(result)
        return result

    async def fetch(
        self,
        source: str,
        identifier: Union[str, int],
        max_chars: int = 16000,
        language: Optional[str] = None,
        site: str = "stackoverflow",
        overall_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fetch article, entity, paper abstract, or Q&A by identifier."""
        eff_lang = self._validate_language(language or self.default_language)
        bound = max(100, min(int(max_chars), 50000))

        coro = self._dedup_run(
            source, "fetch", str(identifier), 1, eff_lang, site
        )
        if overall_timeout:
            raw_result = await asyncio.wait_for(coro, timeout=overall_timeout)
        else:
            raw_result = await coro

        result = copy.deepcopy(raw_result)
        for item in result.get("results", []):
            text = item.get("content", "")
            item["truncated"] = len(text) > bound
            item["content"] = text[:bound]

        # Enforce total serialized output bounding
        self._bound_serialized_results(result)
        return result

    def _bound_serialized_results(self, result: Dict[str, Any]):
        """Ensure total serialized size of results does not exceed max_serialized_output_chars."""
        try:
            dumped = json.dumps(result, ensure_ascii=False)
            if len(dumped) <= self.max_serialized_output_chars:
                return

            # Shorten answers list in Stack Exchange or claims in Wikidata if needed
            for item in result.get("results", []):
                if "answers" in item and isinstance(item["answers"], list) and len(item["answers"]) > 2:
                    item["answers"] = item["answers"][:2]
                    item["answers_bounded_by_size"] = True
                if "statements" in item and isinstance(item["statements"], dict):
                    # Keep only first 20 properties
                    truncated_claims = dict(list(item["statements"].items())[:20])
                    item["statements"] = truncated_claims
                    item["statements_bounded_by_size"] = True

            dumped = json.dumps(result, ensure_ascii=False)
            if len(dumped) <= self.max_serialized_output_chars:
                return

            # Shorten content fields across results
            for item in result.get("results", []):
                dumped = json.dumps(result, ensure_ascii=False)
                if len(dumped) <= self.max_serialized_output_chars:
                    break
                current_c = item.get("content", "")
                if current_c:
                    excess = len(dumped) - self.max_serialized_output_chars + 10
                    if len(current_c) > excess:
                        item["content"] = current_c[:-excess]
                        item["truncated"] = True
                    else:
                        item["content"] = current_c[:50]
                        item["truncated"] = True

            # If still exceeding (e.g. too many search result items), truncate result list
            dumped = json.dumps(result, ensure_ascii=False)
            results_list = result.get("results", [])
            while len(dumped) > self.max_serialized_output_chars and len(results_list) > 1:
                results_list.pop()
                result["results_truncated_by_size"] = True
                dumped = json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Output bounding error: {e}")

    async def _dedup_run(
        self,
        source: str,
        operation: str,
        value: str,
        limit: int,
        language: str,
        site: str,
    ) -> Dict[str, Any]:
        """Single-flight request deduplication across concurrent callers."""
        key = json.dumps([source, operation, value, limit, language, site])
        async with self._inflight_lock:
            if key in self._inflight:
                future = self._inflight[key]
            else:
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._inflight[key] = future
                # Schedule actual operation
                asyncio.create_task(self._execute_and_resolve(key, future, source, operation, value, limit, language, site))

        return await asyncio.shield(future)

    async def _execute_and_resolve(
        self,
        key: str,
        future: asyncio.Future,
        source: str,
        operation: str,
        value: str,
        limit: int,
        language: str,
        site: str,
    ):
        try:
            res = await self._run(source, operation, value, limit, language, site)
            if not future.done():
                future.set_result(res)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
        finally:
            async with self._inflight_lock:
                self._inflight.pop(key, None)

    async def _run(
        self,
        source: str,
        operation: str,
        value: str,
        limit: int,
        language: str,
        site: str,
    ) -> Dict[str, Any]:
        if source not in SOURCES:
            raise ValueError(f"Unknown source '{source}'")
        if language not in ("en", "fr"):
            raise ValueError(f"Unsupported language '{language}'")
        if source == "stackexchange" and site not in SITES:
            raise ValueError(f"Unsupported Stack Exchange site '{site}'")
        if source != "stackexchange" and site not in SITES:
            site = "stackoverflow"

        key = json.dumps([source, operation, value, limit, language, site])
        async with self.locks[source]:
            cached = None
            if self.db:
                try:
                    cached = self.db.execute(
                        "SELECT saved, value FROM knowledge_cache WHERE key = ?", (key,)
                    ).fetchone()
                except Exception as e:
                    logger.debug(f"Cache read error: {e}")

            if cached and (time.time() - cached[0] < self.ttl):
                return dict(json.loads(cached[1]), cached=True, stale=False)

            try:
                if time.monotonic() < self.until.get(source, 0):
                    raise ProviderError("Provider cooldown", "cooldown")

                items = await self._adapter(source, operation, value, limit, language, site)
                output = dict(
                    status="ok" if items else ("no_results" if operation == "search" else "not_found"),
                    source=source,
                    results=items,
                    cached=False,
                    stale=False,
                )

                if self.db:
                    try:
                        self.db.execute(
                            "INSERT OR REPLACE INTO knowledge_cache VALUES (?, ?, ?)",
                            (key, time.time(), json.dumps(output)),
                        )
                        self.db.commit()
                        self._evict_cache_if_needed()
                    except Exception as e:
                        logger.debug(f"Cache write error: {e}")

                return output

            except (ProviderError, httpx.HTTPError, ValueError, ET.ParseError, KeyError, TypeError) as exc:
                status = getattr(exc, "status", "unavailable")
                self.until[source] = max(self.until.get(source, 0), time.monotonic() + self.cooldown)
                if cached and (time.time() - cached[0] < self.stale_ttl):
                    return dict(json.loads(cached[1]), cached=True, stale=True, warning=status)
                return dict(
                    status=status,
                    source=source,
                    results=[],
                    error=type(exc).__name__,
                    cached=False,
                    stale=False,
                )

    async def _get(self, source: str, url: str, **params: Any) -> httpx.Response:
        """Issue HTTP GET with rate limiting, retries, 429 Retry-After, and 4MB limit."""
        for attempt in range(self.retries + 1):
            await asyncio.sleep(max(0.0, self.next_request.get(source, 0.0) - time.monotonic()))
            spacing = self.spacing if source == "arxiv" else 0.2
            try:
                async with self.http.stream("GET", url, params=params) as response:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 4_000_000:
                            raise ProviderError("Response too large")
                    headers = dict(response.headers)
                    # aiter_bytes already decompressed gzip/br/zstd
                    headers.pop("content-encoding", None)
                    headers.pop("content-length", None)
                    response = httpx.Response(
                        response.status_code,
                        headers=headers,
                        content=bytes(body),
                        request=response.request,
                    )
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == self.retries:
                    raise
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            finally:
                self.next_request[source] = time.monotonic() + spacing

            if response.status_code == 429:
                raw = response.headers.get("retry-after", "60")
                try:
                    delay = float(raw)
                except ValueError:
                    try:
                        delay = parsedate_to_datetime(raw).timestamp() - time.time()
                    except (ValueError, TypeError, OverflowError):
                        delay = 60.0
                self.until[source] = time.monotonic() + max(1.0, delay)
                raise ProviderError("Rate limit", "rate_limited")

            if response.status_code >= 500 and attempt < self.retries:
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            response.raise_for_status()
            return response

    async def _json(self, source: str, url: str, **params: Any) -> Dict[str, Any]:
        """Fetch JSON data and handle Stack Exchange backoff/errors."""
        data = (await self._get(source, url, **params)).json()
        if not isinstance(data, dict):
            raise ProviderError("Invalid JSON object")
        if "backoff" in data:
            self.next_request[source] = max(
                self.next_request.get(source, 0.0),
                time.monotonic() + float(data["backoff"]),
            )
        if "error" in data or "error_id" in data:
            raise ProviderError("API error")
        return data

    def _enrich_wikidata_claims(self, claims: Dict[str, Any], lang: str) -> Dict[str, Any]:
        """Add human-readable property labels to Wikidata claims for better context."""
        enriched: Dict[str, Any] = {}
        for prop_id, snaks in claims.items():
            label = WIKIDATA_PROPERTY_LABELS.get(prop_id)
            key = f"{prop_id} ({label})" if label else prop_id
            enriched[key] = snaks
        return enriched

    async def _adapter(
        self,
        source: str,
        op: str,
        value: str,
        limit: int,
        lang: str,
        site: str,
    ) -> List[Dict[str, Any]]:
        # 1. Wikipedia Adapter
        if source == "wikipedia":
            url = f"https://{lang}.wikipedia.org/w/api.php"
            params = dict(action="query", format="json", formatversion=2)
            if op == "search":
                data = await self._json(
                    source,
                    url,
                    **params,
                    list="search",
                    srsearch=value,
                    srnamespace=0,
                    srlimit=limit,
                )
                return [
                    record(
                        source,
                        p["pageid"],
                        p["title"],
                        f"https://{lang}.wikipedia.org/?curid={p['pageid']}",
                        plain(p.get("snippet")),
                        language=lang,
                        content_kind="snippet",
                    )
                    for p in data.get("query", {}).get("search", [])
                ]

            if not value.isdigit():
                raise ValueError("Wikipedia requires numeric page ID")

            data = await self._json(
                source,
                url,
                **params,
                pageids=value,
                prop="extracts|info|pageprops",
                explaintext=1,
                inprop="url",
                ppprop="disambiguation",
            )
            return [
                record(
                    source,
                    p["pageid"],
                    p["title"],
                    p.get("fullurl") or f"https://{lang}.wikipedia.org/?curid={p['pageid']}",
                    p.get("extract", ""),
                    revision_id=p.get("lastrevid"),
                    language=lang,
                    content_kind="article_extract",
                    disambiguation="disambiguation" in p.get("pageprops", {}),
                )
                for p in data.get("query", {}).get("pages", [])
                if "missing" not in p
            ]

        # 2. Wikidata Adapter
        if source == "wikidata":
            url = "https://www.wikidata.org/w/api.php"
            if op == "search":
                data = await self._json(
                    source,
                    url,
                    action="wbsearchentities",
                    format="json",
                    search=value,
                    language=lang,
                    limit=limit,
                    type="item",
                )
                return [
                    record(
                        source,
                        p["id"],
                        p.get("label", p["id"]),
                        f"https://www.wikidata.org/wiki/{p['id']}",
                        p.get("description", ""),
                        licence="CC0",
                        content_kind="entity_description",
                    )
                    for p in data.get("search", [])
                ]

            if not re.fullmatch(r"Q[1-9]\d*", value):
                raise ValueError("Invalid entity ID (must match Q-number e.g. Q381)")

            data = await self._json(
                source,
                url,
                action="wbgetentities",
                format="json",
                ids=value,
                props="labels|descriptions|claims|info",
                languages=lang,
            )
            entities = data.get("entities", {})
            p = entities.get(value, {})
            if "missing" in p or not p:
                return []

            claims = p.get("claims", {})
            enriched_claims = self._enrich_wikidata_claims(claims, lang)
            return [
                record(
                    source,
                    value,
                    p.get("labels", {}).get(lang, {}).get("value", value),
                    f"https://www.wikidata.org/wiki/{value}",
                    json.dumps(claims, ensure_ascii=False),
                    statements=claims,
                    enriched_statements=enriched_claims,
                    revision_id=p.get("lastrevid"),
                    updated_at=p.get("modified"),
                    licence="CC0",
                    content_kind="structured_statements",
                )
            ]

        # 3. arXiv Adapter
        if source == "arxiv":
            if op == "fetch" and not re.fullmatch(
                r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", value
            ):
                raise ValueError("Invalid arXiv ID")

            params = (
                {"search_query": "all:" + value, "max_results": limit}
                if op == "search"
                else {"id_list": value}
            )
            r = await self._get(source, "https://export.arxiv.org/api/query", **params)
            root = ET.fromstring(r.content)
            if root.tag != "{http://www.w3.org/2005/Atom}feed":
                raise ProviderError("Invalid Atom feed")

            items = []
            for p in root.findall("a:entry", ATOM):
                entry_url = p.findtext("a:id", "", ATOM)
                if "/api/errors" in entry_url:
                    raise ProviderError("arXiv query error")
                identifier = entry_url.split("/abs/")[-1]
                items.append(
                    record(
                        source,
                        identifier,
                        " ".join(p.findtext("a:title", "", ATOM).split()),
                        entry_url.replace("http:", "https:", 1),
                        p.findtext("a:summary", "", ATOM).strip(),
                        published_at=p.findtext("a:published", None, ATOM),
                        updated_at=p.findtext("a:updated", None, ATOM),
                        authors=[a.findtext("a:name", "", ATOM) for a in p.findall("a:author", ATOM)],
                        content_kind="abstract",
                        licence="See individual paper licence",
                    )
                )
            return items

        # 4. Stack Exchange Adapter
        base = "https://api.stackexchange.com/2.3"
        if op == "search":
            data = await self._json(
                source,
                base + "/search/advanced",
                site=site,
                q=value,
                pagesize=limit,
                order="desc",
                sort="relevance",
                filter="withbody",
            )
            rows = data.get("items", [])
        else:
            if not value.isdigit():
                raise ValueError("Invalid question ID")
            data = await self._json(
                source,
                base + "/questions/" + value,
                site=site,
                filter="withbody",
            )
            rows = data.get("items", [])
            if rows:
                q_item = rows[0]
                accepted_id = q_item.get("accepted_answer_id")

                # Fetch answers sorted by votes
                answers_data = await self._json(
                    source,
                    base + "/questions/" + value + "/answers",
                    site=site,
                    filter="withbody",
                    sort="votes",
                    order="desc",
                    pagesize=5,
                )
                answer_list = answers_data.get("items", [])

                # Hardening: If accepted answer exists and is not among top 5, fetch it specifically
                if accepted_id and not any(a.get("answer_id") == accepted_id for a in answer_list):
                    try:
                        acc_data = await self._json(
                            source,
                            f"{base}/answers/{accepted_id}",
                            site=site,
                            filter="withbody",
                        )
                        acc_items = acc_data.get("items", [])
                        if acc_items:
                            # Prepend accepted answer so it is never omitted
                            answer_list.insert(0, acc_items[0])
                    except Exception as ae:
                        logger.debug(f"Failed to fetch accepted answer {accepted_id}: {ae}")

                q_item["answers"] = answer_list

        return [
            record(
                source,
                p.get("question_id"),
                p.get("title", ""),
                p.get("link", ""),
                plain(p.get("body", "")),
                site=site,
                score=p.get("score"),
                accepted_answer_id=p.get("accepted_answer_id"),
                author=p.get("owner"),
                published_at=p.get("creation_date"),
                updated_at=p.get("last_activity_date"),
                answers=p.get("answers", []),
                answers_limited=op == "fetch",
                licence=p.get("content_license", "See Stack Exchange attribution requirements"),
                content_kind="question_and_answers" if op == "fetch" else "question",
            )
            for p in rows
        ]

    async def search_and_fetch_best(
        self,
        query: str,
        source: str = "auto",
        language: Optional[str] = None,
        max_chars: Optional[int] = None,
        site: str = "stackoverflow",
        overall_timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Search and fetch the best candidate result across knowledge providers."""
        eff_source = self.route(query) if source == "auto" else source
        search_res = await self.search(
            query,
            source=eff_source,
            limit=5,
            language=language,
            site=site,
            overall_timeout=overall_timeout,
        )
        results = search_res.get("results", [])
        if not results:
            return None

        # Filter out disambiguation pages for Wikipedia if possible
        if eff_source == "wikipedia":
            candidates = [r for r in results if "(disambiguation)" not in r.get("title", "").lower()]
            target = candidates[0] if candidates else results[0]
        else:
            target = results[0]

        fetch_res = await self.fetch(
            eff_source,
            target.get("identifier"),
            max_chars=max_chars or 16000,
            language=language,
            site=site,
            overall_timeout=overall_timeout,
        )
        fetch_items = fetch_res.get("results", [])
        if not fetch_items:
            return None
        item = copy.deepcopy(fetch_items[0])
        item["status"] = fetch_res.get("status", "ok")
        item["text"] = item.get("content", "")
        return item
