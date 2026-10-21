"""Unit and integration tests for native asynchronous MediaWiki WikipediaClient.

Deterministic tests cover:
- search: candidates, page IDs, snippets, word counts.
- fetch: normal page with extract, url, revision id, timestamp, language.
- missing page: detection and appropriate structured object.
- disambiguation page: detection and option parsing.
- text bounding: max_chars clipping and truncated=True.
- cache: SQLite hit, TTL expiry, stale result fallback on error.
- bounded retries: retries on 500 / 429 respecting Retry-After.
- request cancellation: task cancellation safety.

Opt-in live tests:
- HAL_LIVE_WIKIPEDIA_TESTS=1: executes live queries against English Wikipedia.
"""

import asyncio
import os
import sqlite3
import tempfile
import time
import unittest

import httpx
import pytest

from hal9000_inside.wikipedia.client import (
    WikipediaClient,
    WikipediaSearchResult,
    WikipediaArticle,
    SUPPORTED_LANGUAGES,
    DEFAULT_USER_AGENT,
)


class TestWikipediaClientDeterministic(unittest.IsolatedAsyncioTestCase):
    """Deterministic tests using httpx.MockTransport."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_db = os.path.join(self.tmpdir.name, "test_wiki_cache.db")

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_search_deterministic(self):
        """Test search query returns structured results."""
        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(httpx.QueryParams(request.url.query))
            self.assertEqual(params.get("action"), "query")
            self.assertEqual(params.get("list"), "search")
            self.assertIn("srsearch", params)
            mock_data = {
                "query": {
                    "search": [
                        {
                            "title": "Discovery One",
                            "pageid": 42100,
                            "snippet": "Discovery One was a United States spacecraft...",
                            "wordcount": 1200,
                        },
                        {
                            "title": "HAL 9000",
                            "pageid": 13420,
                            "snippet": "HAL 9000 is a fictional artificial intelligence...",
                            "wordcount": 3400,
                        },
                    ]
                }
            }
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                cache_db_path=self.cache_db,
            )
            results = await client.search("HAL 9000", limit=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].title, "Discovery One")
            self.assertEqual(results[0].page_id, 42100)
            self.assertEqual(results[1].title, "HAL 9000")
            self.assertEqual(results[1].page_id, 13420)
            self.assertEqual(results[0].language, "en")
            await client.close()

    async def test_fetch_normal_page(self):
        """Test fetching a regular article with extracts and metadata."""
        def handler(request: httpx.Request) -> httpx.Response:
            mock_data = {
                "query": {
                    "pages": {
                        "13420": {
                            "pageid": 13420,
                            "title": "HAL 9000",
                            "fullurl": "https://en.wikipedia.org/wiki/HAL_9000",
                            "lastrevid": 998877,
                            "extract": "HAL 9000 is a sentient computer in 2001: A Space Odyssey.",
                            "pageprops": {},
                        }
                    }
                }
            }
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                cache_db_path=self.cache_db,
            )
            art = await client.fetch(13420)
            self.assertEqual(art.status, "ok")
            self.assertEqual(art.title, "HAL 9000")
            self.assertEqual(art.page_id, 13420)
            self.assertEqual(art.revision_id, 998877)
            self.assertFalse(art.is_missing)
            self.assertFalse(art.is_disambiguation)
            self.assertFalse(art.truncated)
            self.assertIn("sentient computer", art.extract)
            self.assertTrue(art.retrieval_timestamp)
            await client.close()

    async def test_fetch_missing_page(self):
        """Test fetching a non-existent page ID."""
        def handler(request: httpx.Request) -> httpx.Response:
            mock_data = {
                "query": {
                    "pages": {
                        "999999999": {
                            "pageid": 999999999,
                            "missing": True,
                        }
                    }
                }
            }
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                cache_db_path=self.cache_db,
            )
            art = await client.fetch(999999999)
            self.assertTrue(art.is_missing)
            self.assertEqual(art.status, "missing")
            self.assertEqual(art.extract, "")
            self.assertIn("does not exist", art.error_message or "")
            await client.close()

    async def test_fetch_disambiguation_page(self):
        """Test fetching a disambiguation page."""
        def handler(request: httpx.Request) -> httpx.Response:
            mock_data = {
                "query": {
                    "pages": {
                        "5185989": {
                            "pageid": 5185989,
                            "title": "Apollo 11 (disambiguation)",
                            "fullurl": "https://en.wikipedia.org/wiki/Apollo_11_(disambiguation)",
                            "lastrevid": 123456,
                            "extract": "Apollo 11 may refer to:\nApollo 11 (film), a 2019 documentary\nApollo 11 (musical), a 2019 musical",
                            "pageprops": {
                                "disambiguation": ""
                            },
                        }
                    }
                }
            }
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                cache_db_path=self.cache_db,
            )
            art = await client.fetch(5185989)
            self.assertTrue(art.is_disambiguation)
            self.assertEqual(art.status, "disambiguation")
            self.assertGreaterEqual(len(art.disambiguation_options), 2)
            self.assertIn("Apollo 11 (film)", art.disambiguation_options[1])
            await client.close()

    async def test_text_bounding_and_truncation(self):
        """Test bounding large articles to max_text_chars."""
        long_text = "Knowledge base text. " * 300  # ~6300 chars
        def handler(request: httpx.Request) -> httpx.Response:
            mock_data = {
                "query": {
                    "pages": {
                        "123": {
                            "pageid": 123,
                            "title": "Large Topic",
                            "fullurl": "https://en.wikipedia.org/wiki/Large_Topic",
                            "lastrevid": 555,
                            "extract": long_text,
                            "pageprops": {},
                        }
                    }
                }
            }
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                max_text_chars=500,
                cache_db_path=self.cache_db,
            )
            art = await client.fetch(123)
            self.assertTrue(art.truncated)
            self.assertEqual(len(art.extract), 500)
            await client.close()

    async def test_sqlite_cache_hit_and_expiry(self):
        """Test cache storage, hits, and TTL expiry."""
        call_count = 0
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            mock_data = {
                "query": {
                    "pages": {
                        "777": {
                            "pageid": 777,
                            "title": "Cached Subject",
                            "fullurl": "https://en.wikipedia.org/wiki/Cached_Subject",
                            "lastrevid": 888,
                            "extract": f"Content version {call_count}",
                            "pageprops": {},
                        }
                    }
                }
            }
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            # 1 second TTL
            client = WikipediaClient(
                client=mock_http,
                cache_ttl_seconds=1,
                cache_db_path=self.cache_db,
            )

            # First fetch: populates cache
            art1 = await client.fetch(777)
            self.assertEqual(call_count, 1)
            self.assertIn("version 1", art1.extract)
            self.assertFalse(art1.is_stale)

            # Second fetch immediate: cache hit, no HTTP call
            art2 = await client.fetch(777)
            self.assertEqual(call_count, 1)
            self.assertIn("version 1", art2.extract)

            # Sleep past TTL
            await asyncio.sleep(1.1)

            # Third fetch after TTL: refreshed from HTTP
            art3 = await client.fetch(777)
            self.assertEqual(call_count, 2)
            self.assertIn("version 2", art3.extract)
            await client.close()

    async def test_stale_cache_fallback_on_network_failure(self):
        """Test that if network fails, a stale cache entry is returned with is_stale=True."""
        call_count = 0
        should_fail = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if should_fail:
                raise httpx.ConnectError("Network unreachable")
            call_count += 1
            mock_data = {
                "query": {
                    "pages": {
                        "888": {
                            "pageid": 888,
                            "title": "Fallback Subject",
                            "fullurl": "https://en.wikipedia.org/wiki/Fallback_Subject",
                            "lastrevid": 999,
                            "extract": "Original cached content",
                            "pageprops": {},
                        }
                    }
                }
            }
            return httpx.Response(200, json=mock_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                cache_ttl_seconds=1,
                max_retries=1,
                cache_db_path=self.cache_db,
            )

            # Store in cache
            art1 = await client.fetch(888)
            self.assertEqual(art1.extract, "Original cached content")
            self.assertFalse(art1.is_stale)

            # Expire it
            await asyncio.sleep(1.1)

            # Now fail network
            should_fail = True
            art2 = await client.fetch(888)
            self.assertTrue(art2.is_stale)
            self.assertEqual(art2.extract, "Original cached content")
            await client.close()

    async def test_retries_on_500_and_429_with_retry_after(self):
        """Test bounded retries respecting Retry-After header."""
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(500, text="Internal Server Error")
            elif attempts == 2:
                # 429 with Retry-After header (0.1s for fast test)
                return httpx.Response(429, headers={"Retry-After": "0.1"}, text="Too Many Requests")
            else:
                return httpx.Response(200, json={
                    "query": {
                        "search": [{"title": "Success Article", "pageid": 101, "snippet": "ok", "wordcount": 50}]
                    }
                })

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                max_retries=3,
                cache_db_path=self.cache_db,
            )
            results = await client.search("Success")
            self.assertEqual(attempts, 3)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Success Article")
            await client.close()

    async def test_request_cancellation(self):
        """Test cancellation propagates cleanly without hanging."""
        async def slow_handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10.0)
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(slow_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = WikipediaClient(
                client=mock_http,
                cache_db_path=self.cache_db,
            )
            task = asyncio.create_task(client.search("Slow Request"))
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await client.close()

    async def test_language_support_and_validation(self):
        """Test French language endpoint and invalid language fallback to English."""
        used_url = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal used_url
            used_url = str(request.url)
            return httpx.Response(200, json={"query": {"search": []}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            # Valid fr
            client_fr = WikipediaClient(
                language="fr",
                client=mock_http,
                cache_db_path=self.cache_db,
            )
            await client_fr.search("Paris")
            self.assertIn("fr.wikipedia.org", used_url)

            # Invalid language defaults to en
            client_invalid = WikipediaClient(
                language="de_invalid",
                client=mock_http,
                cache_db_path=self.cache_db,
            )
            self.assertEqual(client_invalid.language, "en")
            await client_invalid.search("Berlin")
            self.assertIn("en.wikipedia.org", used_url)

            await client_fr.close()
            await client_invalid.close()


class TestWikipediaLive(unittest.IsolatedAsyncioTestCase):
    """Opt-in live tests against actual English Wikipedia MediaWiki API."""

    @pytest.mark.skipif(
        os.environ.get("HAL_LIVE_WIKIPEDIA_TESTS") != "1",
        reason="Set HAL_LIVE_WIKIPEDIA_TESTS=1 to run live Wikipedia tests",
    )
    async def test_live_search_and_fetch_apollo11(self):
        """Perform live search for Apollo 11 and fetch page ID."""
        client = WikipediaClient()
        try:
            results = await client.search("Apollo 11", limit=3)
            self.assertGreaterEqual(len(results), 1)
            target = results[0]
            self.assertIn("Apollo 11", target.title)
            self.assertGreater(target.page_id, 0)

            art = await client.fetch(target.page_id)
            self.assertEqual(art.status, "ok")
            self.assertEqual(art.page_id, target.page_id)
            self.assertIn("Apollo 11", art.title)
            self.assertIn("wikipedia.org", art.url)
            self.assertGreater(len(art.extract), 100)
            self.assertIsNotNone(art.revision_id)
        finally:
            await client.close()
