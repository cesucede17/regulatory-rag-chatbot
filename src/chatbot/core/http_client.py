"""HTTP client for BOE sumario API with retry logic."""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import httpx

from ..config import settings


class BOESumarioClient:
    """HTTP client for BOE sumario API."""

    def __init__(self) -> None:
        self.base_url = settings.boe_sumario_api_url
        self.max_retries = 3
        self.retry_interval = 0.5  # seconds
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl: float = 3600.0  # 1 hour

    def fetch_sumario(self, date: str) -> Optional[Dict[str, Any]]:
        """
        Fetch BOE sumario for a specific date.

        Args:
            date: Date in YYYYMMDD format

        Returns:
            JSON response from BOE API, or None on failure
        """
        url = f"{self.base_url}{date}"

        # Return cached response if still fresh
        cached = self._cache.get(url)
        if cached:
            data, ts = cached
            if time.time() - ts < self._cache_ttl:
                return data

        headers = {"Accept": "application/json"}

        with httpx.Client(verify=True) as client:
            for attempt in range(self.max_retries):
                try:
                    response = client.get(url, headers=headers, timeout=30.0)
                    response.raise_for_status()
                    result = response.json()
                    self._cache[url] = (result, time.time())
                    return result
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    # Sumario for a date may not exist; do not retry on 404.
                    if status_code == 404:
                        return None
                    if attempt < self.max_retries - 1 and status_code >= 500:
                        time.sleep(self.retry_interval)
                        continue
                    print(
                        f"HTTP status error {status_code} on attempt "
                        f"{attempt + 1}/{self.max_retries}: {exc}"
                    )
                    return None
                except httpx.HTTPError as exc:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_interval)
                        continue
                    print(
                        f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {exc}"
                    )
                    return None
                except Exception as exc:
                    print(f"Unexpected error: {exc}")
                    return None

        return None

    def fetch_document_text(self, doc_id: str) -> Optional[str]:
        """
        Fetch the full text of a BOE document by its identifier (e.g. BOE-A-2023-2027).

        Uses the BOE XML API, strips tags, and returns plain text truncated to
        a size suitable for LLM context.
        """
        import re as _re

        url = f"https://www.boe.es/diario_boe/xml.php?id={doc_id}"
        try:
            with httpx.Client(verify=True) as client:
                r = client.get(url, timeout=30.0)
                r.raise_for_status()
                text = _re.sub(r"<[^>]+>", " ", r.text)
                text = _re.sub(r"\s+", " ", text).strip()
                return text[:60000]
        except Exception as exc:
            print(f"Error fetching BOE document {doc_id}: {exc}")
            return None

    def fetch_latest_sumario(
        self,
        days_back: int = 7,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetch the most recent available BOE sumario, starting from today.

        Returns:
            Tuple of (sumario_json, yyyymmdd_date) or (None, None)
        """
        for offset in range(max(days_back, 1)):
            candidate_date = (datetime.now() - timedelta(days=offset)).strftime(
                "%Y%m%d"
            )
            data = self.fetch_sumario(candidate_date)
            if data:
                return data, candidate_date
        return None, None
