"""
search_tool.py - Tavily Search API Integration for RESEARCHLY Agent.

Features:
- Connect & response timeout controls
- Query deduplication & in-memory cache
- Snippet truncation for prompt optimization
- Safe diagnostic logging with elapsed time
- Handles errors, network timeouts, empty results, and missing keys.
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel

logger = logging.getLogger("researchly.search")

MAX_SNIPPET_LENGTH = 1000  # Truncate each source snippet to 1000 chars to optimize LLM tokens


class SearchResult(BaseModel):
    id: int
    title: str
    url: str
    content: str
    score: float = 0.0
    domain: str = ""


class SearchToolError(Exception):
    """Custom exception raised when search tool encounters an error."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class TavilySearchTool:
    """Wrapper around Tavily Search API with caching and timeout optimization."""

    # In-memory query cache: {normalized_query: (timestamp, [SearchResult])}
    _cache: Dict[str, Tuple[float, List[SearchResult]]] = {}
    CACHE_TTL_SECONDS = 600  # 10 minutes

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        connect_timeout: float = 5.0
    ):
        self.api_key = api_key.strip() if api_key is not None else os.getenv("TAVILY_API_KEY", "").strip()
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.api_url = "https://api.tavily.com/search"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def clear_cache(cls):
        """Clears the in-memory cache."""
        cls._cache.clear()

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc or url
        except Exception:
            return url

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Executes web search for query and returns up to max_results structured results.
        
        Args:
            query: The research query string.
            max_results: Max number of sources to return (1-5).
        
        Returns:
            List of SearchResult objects with id, title, url, content, score, domain.
        """
        clean_query = query.strip()
        if not clean_query:
            raise SearchToolError("Search query cannot be empty.")

        if not self.api_key:
            raise SearchToolError(
                "TAVILY_API_KEY is not configured. Please set the environment variable."
            )

        # Check in-memory cache to prevent duplicate queries
        cache_key = f"{clean_query.lower()}:{max_results}"
        now = time.time()
        if cache_key in self._cache:
            cached_time, cached_results = self._cache[cache_key]
            if now - cached_time < self.CACHE_TTL_SECONDS:
                logger.info(f"[SEARCH CACHE HIT] Reusing {len(cached_results)} cached results for query: '{clean_query[:50]}'")
                return cached_results

        clamped_results = min(max(max_results, 1), 5)
        payload = {
            "api_key": self.api_key,
            "query": clean_query,
            "search_depth": "basic",
            "max_results": clamped_results,
            "include_answer": False,
            "include_raw_content": False
        }

        # DIAGNOSTIC LOG: Search started
        start_time = time.perf_counter()
        logger.info(f"[STAGE 2: SEARCH STARTED] Query: '{clean_query[:60]}', max_results: {clamped_results}")

        timeout_config = httpx.Timeout(self.timeout, connect=self.connect_timeout)

        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(self.api_url, json=payload)
        except httpx.ConnectTimeout:
            elapsed = time.perf_counter() - start_time
            logger.error(f"[SEARCH ERROR] Connection timed out after {elapsed:.2f}s (connect deadline: {self.connect_timeout}s)")
            raise SearchToolError(f"Connection to search service timed out after {self.connect_timeout}s.")
        except httpx.TimeoutException:
            elapsed = time.perf_counter() - start_time
            logger.error(f"[SEARCH ERROR] Request timed out after {elapsed:.2f}s (total deadline: {self.timeout}s)")
            raise SearchToolError(f"Search request timed out after {self.timeout}s.")
        except httpx.RequestError as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(f"[SEARCH ERROR] Network error after {elapsed:.2f}s: {exc}")
            raise SearchToolError(f"Network error while connecting to search service: {str(exc)}")

        elapsed = time.perf_counter() - start_time

        if response.status_code != 200:
            error_detail = response.text[:200]
            logger.error(f"[SEARCH ERROR] Tavily returned status {response.status_code} in {elapsed:.2f}s: {error_detail}")
            raise SearchToolError(
                f"Tavily search API returned status {response.status_code}: {error_detail}",
                status_code=response.status_code
            )

        try:
            data = response.json()
        except Exception:
            logger.error(f"[SEARCH ERROR] Invalid JSON received from Tavily in {elapsed:.2f}s")
            raise SearchToolError("Invalid response JSON from search service.")

        raw_results = data.get("results", [])

        # DIAGNOSTIC LOG: Search completed
        logger.info(
            f"[STAGE 3: SEARCH COMPLETED] Elapsed: {elapsed:.2f}s, Raw results: {len(raw_results)}"
        )

        if not raw_results:
            return []

        structured_results: List[SearchResult] = []
        for idx, item in enumerate(raw_results[:clamped_results], start=1):
            title = (item.get("title") or "Untitled Source").strip()
            url = (item.get("url") or "").strip()
            content = (item.get("content") or "").strip()
            raw_score = item.get("score", 0.0)
            try:
                score = round(float(raw_score), 4)
            except (ValueError, TypeError):
                score = 0.0

            domain = self._extract_domain(url)

            # Sanitize content snippets to remove excessive whitespace and clamp length
            clean_content = re.sub(r"\s+", " ", content)
            if len(clean_content) > MAX_SNIPPET_LENGTH:
                clean_content = clean_content[:MAX_SNIPPET_LENGTH].rstrip() + "..."

            structured_results.append(
                SearchResult(
                    id=idx,
                    title=title,
                    url=url,
                    content=clean_content,
                    score=score,
                    domain=domain
                )
            )

        # Store in cache
        self._cache[cache_key] = (now, structured_results)
        return structured_results
