"""Download INTERMAGNET 1-minute data (IAGA-2002) and the GFZ Kp index.

Files are cached under data/raw, so a re-run only fetches what is still missing.
Network failures are retried with backoff; a chunk that still fails is reported and
skipped instead of aborting the whole download.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import config

USER_AGENT = "magfault/0.1 (open-source research project)"

# Dropped connections, truncated chunked responses, timeouts, DNS hiccups.
RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)

# Fallbacks if the GFZ JSON service is unreachable: the daily-updated ASCII file.
KP_TEXT_URLS = (
    "https://kp.gfz-potsdam.de/app/files/Kp_ap_since_1932.txt",
    "https://kp.gfz.de/app/files/Kp_ap_since_1932.txt",
    "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt",
)


def chunks(start: date, end: date, max_days: int = 16) -> Iterator[tuple[date, date]]:
    """Yield (first_day, last_day) pieces that never cross a month boundary and are
    at most max_days long, clipped to [start, end]."""
    cur = start
    while cur <= end:
        month_end = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1) - timedelta(days=1)
        last = min(cur + timedelta(days=max_days - 1), month_end, end)
        yield cur, last
        cur = last + timedelta(days=1)


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


def _fetch(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except RETRYABLE as exc:
            if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                raise                                   # a bad request will not fix itself
            if attempt == retries:
                raise
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"  retry {attempt}/{retries - 1} in {wait}s ({type(exc).__name__}: {exc})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def download_intermagnet(stations: list[str], start: date, end: date, out: Path,
                         force: bool = False, max_days: int = 16) -> list[Path]:
    written, failed = [], []
    for code in stations:
        code = code.upper()
        for first, last in chunks(start, end, max_days):
            path = out / code / f"{code}_{first:%Y%m%d}_{last:%Y%m%d}.min"
            if path.exists() and not force:
                continue
            print(f"GET {code} {first}..{last}")
            try:
                body = _fetch(gin_url(code, first, last))
            except RETRYABLE as exc:
                print(f"  FAILED, skipping for now ({type(exc).__name__}: {exc})")
                failed.append(f"{code} {first}..{last}")
                continue
            if b"IAGA-2002" not in body[:400]:
                print(f"  skipped: service returned no IAGA-2002 data ({body[:120]!r})")
                failed.append(f"{code} {first}..{last} (no data)")
                continue
            _write_atomic(path, body)
            written.append(path)
    if failed:
        print(f"\n{len(failed)} chunk(s) not downloaded:")
        for f in failed:
            print(f"  {f}")
        print("Run the same command again to retry only the missing chunks.")
    return written


# ----------------------------------------------------------------------------- Kp

def parse_kp_text(text: str, start: date, end: date) -> dict:
    """Parse GFZ 'Kp_ap_since_1932.txt' rows
    (YYYY MM DD hh.h hh._m days days_m Kp ap D) into the GFZ JSON layout."""
    lo = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    hi = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    times, kps, status = [], [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) < 8:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            hour = float(parts[3])
            kp = float(parts[7])
        except ValueError:
            continue
        t = datetime(y, m, d, tzinfo=timezone.utc) + timedelta(hours=hour)
        if not (lo <= t < hi) or not (0 <= kp <= 9):
            continue
        times.append(t.strftime("%Y-%m-%dT%H:%M:%SZ"))
        kps.append(kp)
        flag = parts[9] if len(parts) > 9 else ""
        status.append("def" if flag == "2" else "pre" if flag in ("0", "1") else None)
    if not times:
        raise ValueError("no Kp rows in the requested date range")
    return {"meta": {"source": "GFZ Potsdam Kp_ap_since_1932.txt (CC BY 4.0)"},
            "datetime": times, "Kp": kps, "status": status}


def _kp_from_payload(raw: bytes, start: date, end: date) -> dict:
    """Accept either the GFZ JSON response or the ASCII since-1932 file."""
    text = raw.decode("utf-8", errors="replace").lstrip()
    if text.startswith("{"):
        payload = json.loads(text)
        if "datetime" not in payload or "Kp" not in payload:
            raise ValueError(f"Unexpected Kp JSON: {text[:200]}")
        return payload
    return parse_kp_text(text, start, end)


def download_kp(start: date, end: date, out: Path, force: bool = False,
                from_file: Path | None = None) -> Path:
    path = out / f"kp_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if path.exists() and not force and from_file is None:
        return path

    if from_file is not None:
        payload = _kp_from_payload(Path(from_file).read_bytes(), start, end)
    else:
        payload, errors = None, []
        print(f"GET Kp {start}..{end} (GFZ JSON service)")
        try:
            payload = _kp_from_payload(_fetch(kp_url(start, end), retries=3), start, end)
        except (*RETRYABLE, ValueError) as exc:
            errors.append(f"{config.KP_URL}: {type(exc).__name__}: {exc}")
        for url in KP_TEXT_URLS if payload is None else ():
            print(f"GET Kp fallback {url}")
            try:
                payload = _kp_from_payload(_fetch(url, retries=2, timeout=120), start, end)
                break
            except (*RETRYABLE, ValueError) as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        if payload is None:
            raise SystemExit(
                "Could not download Kp from any GFZ source:\n  " + "\n  ".join(errors)
                + "\n\nDownload 'Kp_ap_since_1932.txt' in a browser from https://kp.gfz.de/en/data "
                  "and run:\n  magfault kp --start {0} --end {1} --from-file <path to the file>"
                  .format(start, end))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    print(f"{len(payload['Kp'])} Kp values -> {path}")
    return path
