"""Runtime configuration. Everything has a default; the DSN can be overridden by env var."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "sql"
REPORTS_DIR = REPO_ROOT / "reports"

DEFAULT_DSN = "postgresql://magfault:magfault@localhost:5432/magfault"

# Canadian INTERMAGNET observatories spanning ~50 degrees of longitude.
DEFAULT_STATIONS = ("VIC", "MEA", "BRD", "OTT")

# INTERMAGNET GIN (Edinburgh) web service and GFZ Kp JSON service.
GIN_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
KP_URL = "https://kp.gfz.de/app/json/"

# IAGA-2002 sentinels: 99999 = missing, 88888 = element not recorded.
IAGA_MISSING_THRESHOLD = 88887.0

# Kp at or above this marks a geomagnetic storm (NOAA G1) in the ground truth.
STORM_KP = 5.0


def dsn() -> str:
    return os.environ.get("MAGFAULT_DSN", DEFAULT_DSN)
