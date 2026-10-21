"""Download INTERMAGNET 1-minute data (IAGA-2002) and GFZ Kp (JSON).

Files are cached under data/raw so a re-run only fetches what is missing.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

from . import config

USER_AGENT = "magfault/0.1 (open-source research project)"


def month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield (first_day, last_day) per calendar month, clipped to [start, end]."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


def gin_url(code: str, first: date, last: date) -> str:
    params = {
        "request": "GetData",
        "observatoryIagaCode": code,
        "dataStartDate": f"{first:%Y-%m-%d}T00:00:00Z",
        "dataEndDate": f"{last:%Y-%m-%d}T23:59:59Z",
        "Format": "iaga2002",
        "elements": "",
        "publicationState": "adj-or-rep",
        "samplesPerDay": "minute",
    }
    return f"{config.GIN_URL}?{urllib.parse.urlencode(params)}"


def kp_url(first: date, last: date) -> str:
    params = {
        "start": f"{first:%Y-%m-%d}T00:00:00Z",
        "end": f"{last:%Y-%m-%d}T23:59:59Z",
        "index": "Kp",
    }
    return f"{config.KP_URL}?{urllib.parse.urlencode(params)}"


def _fetch(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"  retry {attempt}/{retries} in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_intermagnet(stations: list[str], start: date, end: date, out: Path,
                         force: bool = False) -> list[Path]:
    written = []
    for code in stations:
        code = code.upper()
        for first, last in month_chunks(start, end):
            path = out / code / f"{code}_{first:%Y%m%d}_{last:%Y%m%d}.min"
            if path.exists() and not force:
                continue
            print(f"GET {code} {first}..{last}")
            body = _fetch(gin_url(code, first, last)).decode("utf-8", errors="replace")
            if "IAGA-2002" not in body[:400]:
                print(f"  skipped: service returned no IAGA-2002 data ({body[:120]!r})")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            written.append(path)
    return written


def download_kp(start: date, end: date, out: Path, force: bool = False) -> Path:
    path = out / f"kp_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if path.exists() and not force:
        return path
    print(f"GET Kp {start}..{end}")
    body = _fetch(kp_url(start, end))
    payload = json.loads(body)
    if "datetime" not in payload or "Kp" not in payload:
        raise RuntimeError(f"Unexpected Kp response: {str(payload)[:200]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path
