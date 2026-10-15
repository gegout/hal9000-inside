import asyncio
import json
import os
import tempfile
import unittest
import httpx
import pytest

from hal9000_inside.wikipedia.knowledge import KnowledgeClient, SOURCES, SITES, plain

class TestKnowledgePrototype(unittest.IsolatedAsyncioTestCase):
    async def make(self, handler, **kw):
        kw.setdefault("cache_path", ":memory:")
        c = KnowledgeClient(transport=httpx.MockTransport(handler), arxiv_spacing=0, **kw)
        self.addAsyncCleanup(c.close)
        return c

    async def test_wikipedia_search_fetch_cache(self):
        calls = []
        def handler(r):
            calls.append(r)
            if r.url.params.get('list'):
                return httpx.Response(200, json={'query': {'search': [{'pageid': 1, 'title': 'Test', 'snippet': '<b>text</b>'}]}})
            return httpx.Response(200, json={'query': {'pages': [{'pageid': 1, 'title': 'Test', 'fullurl': 'https://en.wikipedia.org/wiki/Test', 'extract': 'x' * 500, 'lastrevid': 7, 'pageprops': {'disambiguation': ''}}]}})
        c = await self.make(handler)
        a = await c.search('Test', source='wikipedia')
        self.assertEqual(a['results'][0]['content'], 'text')
        self.assertTrue((await c.search('Test', source='wikipedia'))['cached'])
        b = await c.fetch('wikipedia', '1', max_chars=100)
        self.assertTrue(b['results'][0]['truncated'])
        self.assertTrue(b['results'][0]['disambiguation'])
        self.assertEqual(len(calls), 2)

    async def test_wikidata_retains_qualifiers_references(self):
        claims = {'P1': [{'qualifiers': {'P2': [{'snaktype': 'somevalue'}]}, 'references': [{'hash': 'abc'}]}]}
        c = await self.make(lambda r: httpx.Response(200, json={'entities': {'Q1': {'labels': {'en': {'value': 'Entity'}}, 'claims': claims}}}))
        r = await c.fetch('wikidata', 'Q1')
        self.assertEqual(r['results'][0]['statements'], claims)
        self.assertFalse(r['results'][0]['verified'])

    async def test_arxiv_atom(self):
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1234.56789v1</id><title> A paper </title><summary>Abstract</summary><author><name>A</name></author></entry></feed>'
        c = await self.make(lambda r: httpx.Response(200, text=xml))
        r = await c.search('paper', source='arxiv')
        self.assertEqual(r['results'][0]['content_kind'], 'abstract')
        self.assertEqual(r['results'][0]['authors'], ['A'])

    async def test_stackexchange_answers_and_backoff(self):
        def handler(r):
            if r.url.path.endswith('answers'):
                return httpx.Response(200, json={'items': [{'answer_id': 2, 'body': 'Answer', 'is_accepted': True}]})
            return httpx.Response(200, json={'backoff': 0, 'items': [{'question_id': 1, 'title': 'Q', 'link': 'https://stackoverflow.com/q/1', 'body': '<p>Question</p>'}]})
        c = await self.make(handler)
        r = await c.fetch('stackexchange', '1')
        self.assertTrue(r['results'][0]['answers'][0]['is_accepted'])

    async def test_429_cooldown(self):
        calls = []
        def handler(r):
            calls.append(r)
            return httpx.Response(429, headers={'Retry-After': '120'})
        c = await self.make(handler)
        self.assertEqual((await c.search('a', source='wikipedia'))['status'], 'rate_limited')
        self.assertEqual((await c.search('b', source='wikipedia'))['status'], 'cooldown')
        self.assertEqual(len(calls), 1)

    async def test_stale_fallback(self):
        n = 0
        def handler(r):
            nonlocal n
            n += 1
            if n > 1: return httpx.Response(503)
            return httpx.Response(200, json={'query': {'search': []}})
        c = await self.make(handler, ttl=0, retries=0)
        await c.search('x', source='wikipedia')
        r = await c.search('x', source='wikipedia')
        self.assertTrue(r['stale'])
        self.assertEqual(r['warning'], 'unavailable')

    async def test_retry_5xx(self):
        n = 0
        def handler(r):
            nonlocal n
            n += 1
            return httpx.Response(503) if n == 1 else httpx.Response(200, json={'query': {'search': []}})
        c = await self.make(handler)
        self.assertEqual((await c.search('x', source='wikipedia'))['status'], 'no_results')
        self.assertEqual(n, 2)

    async def test_api_error_and_missing(self):
        c = await self.make(lambda r: httpx.Response(200, json={'error': {'code': 'bad'}}))
        self.assertEqual((await c.search('x', source='wikipedia'))['status'], 'unavailable')
        d = await self.make(lambda r: httpx.Response(200, json={'query': {'pages': [{'missing': True}]}}))
        self.assertEqual((await d.fetch('wikipedia', '1'))['status'], 'not_found')

    async def test_malformed(self):
        c = await self.make(lambda r: httpx.Response(200, text='<html>blocked</html>'))
        self.assertEqual((await c.search('x', source='wikidata'))['status'], 'unavailable')
        self.assertEqual((await c.search('x', source='arxiv'))['status'], 'unavailable')

    async def test_cancel_and_routing(self):
        started = asyncio.Event()
        async def handler(r):
            started.set()
            await asyncio.Event().wait()
        c = await self.make(handler)
        t = asyncio.create_task(c.search('x', source='wikipedia'))
        await started.wait()
        t.cancel()
        with self.assertRaises(asyncio.CancelledError): await t
        self.assertEqual(c.route('Ubuntu traceback'), 'stackexchange')
        self.assertEqual(c.route('AI research paper'), 'arxiv')
        self.assertEqual(c.route('date of birth'), 'wikidata')
        self.assertEqual(c.route('memory consolidation'), 'wikipedia')

    async def test_gzip_response(self):
        import gzip
        body = gzip.compress(json.dumps({'query': {'search': []}}).encode())
        c = await self.make(lambda r: httpx.Response(200, content=body, headers={'Content-Encoding': 'gzip'}))
        self.assertEqual((await c.search('x', source='wikipedia'))['status'], 'no_results')

    async def test_validation(self):
        c = await self.make(lambda r: httpx.Response(500))
        with self.assertRaises(ValueError): await c.search('')
        with self.assertRaises(ValueError): await c.search('x', source='unknown')
        self.assertEqual((await c.fetch('wikidata', 'https://evil.invalid'))['status'], 'unavailable')


class TestKnowledgeIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_search_and_fetch_best_helper(self):
        def handler(r):
            if 'list=search' in str(r.url):
                return httpx.Response(200, json={'query': {'search': [{'pageid': 42, 'title': 'Arthur C. Clarke', 'snippet': 'Sci-fi author'}]}})
            elif 'prop=extracts' in str(r.url):
                return httpx.Response(200, json={'query': {'pages': [{'pageid': 42, 'title': 'Arthur C. Clarke', 'fullurl': 'https://en.wikipedia.org/wiki/Arthur_C._Clarke', 'extract': 'Sir Arthur Charles Clarke was an English science-fiction writer...', 'lastrevid': 123456}]}})
            return httpx.Response(404)

        c = KnowledgeClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(c.close)
        best = await c.search_and_fetch_best("Arthur C. Clarke")
        self.assertIsNotNone(best)
        self.assertEqual(best["status"], "ok")
        self.assertEqual(best["title"], "Arthur C. Clarke")
        self.assertEqual(best["source"], "wikipedia")
        self.assertIn("English science-fiction writer", best["text"])
        self.assertFalse(best["verified"])

    async def test_output_bounding(self):
        def handler(r):
            return httpx.Response(200, json={'query': {'search': [{'pageid': 1, 'title': 'Long', 'snippet': 'A' * 2000}] * 10}})

        c = KnowledgeClient(transport=httpx.MockTransport(handler), max_output_chars=500)
        self.addAsyncCleanup(c.close)
        res = await c.search("test", source="wikipedia")
        total_len = len(json.dumps(res))
        self.assertLessEqual(total_len, 600)  # bounded tightly around limit

    async def test_single_flight_deduplication(self):
        req_count = 0
        def handler(r):
            nonlocal req_count
            req_count += 1
            return httpx.Response(200, json={'query': {'search': [{'pageid': 1, 'title': 'Dedup', 'snippet': 'One call'}]}})

        c = KnowledgeClient(transport=httpx.MockTransport(handler), cache_path=":memory:")
        self.addAsyncCleanup(c.close)
        r1, r2 = await asyncio.gather(
            c.search("same-query", source="wikipedia"),
            c.search("same-query", source="wikipedia"),
        )
        self.assertEqual(r1["results"], r2["results"])
        self.assertEqual(req_count, 1)

    async def test_cache_eviction_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cache.sqlite3")
            def handler(r):
                return httpx.Response(200, json={'query': {'search': [{'pageid': 1, 'title': 'Evict', 'snippet': 'X' * 500}]}})
            c = KnowledgeClient(transport=httpx.MockTransport(handler), cache_path=db_path)
            self.addAsyncCleanup(c.close)
            await c.search("test1", source="wikipedia")
            c._evict_cache_if_needed(max_entries=1)
            # Eviction executes without errors
            self.assertTrue(os.path.exists(db_path))
