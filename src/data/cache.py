"""Local disk cache: Parquet for frames, SQLite for the fetch log (incl. negative results)."""

from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class DiskCache:
    def __init__(self, root: Path, namespace: str) -> None:
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "fetch_log.sqlite"
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS fetch_log ("
                " dataset TEXT, key TEXT, status TEXT, detail TEXT, fetched_at TEXT,"
                " PRIMARY KEY (dataset, key))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=30)

    def _path(self, dataset: str, key: str) -> Path:
        d = self.root / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{_safe(key)}.parquet"

    def status(self, dataset: str, key: str) -> str | None:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT status FROM fetch_log WHERE dataset=? AND key=?", (dataset, key)
            ).fetchone()
        return row[0] if row else None

    def read(self, dataset: str, key: str) -> pd.DataFrame | None:
        p = self._path(dataset, key)
        return pd.read_parquet(p) if p.exists() else None

    def write(self, dataset: str, key: str, df: pd.DataFrame | None, status: str = "ok", detail: str = "") -> None:
        if df is not None and status == "ok":
            df.to_parquet(self._path(dataset, key))
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)",
                (dataset, key, status, detail[:500], now),
            )

    def summary(self) -> pd.DataFrame:
        with self._lock, self._connect() as con:
            return pd.read_sql("SELECT dataset, status, COUNT(*) n FROM fetch_log GROUP BY 1,2", con)
