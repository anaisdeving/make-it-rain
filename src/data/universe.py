"""Candidate universe from public S&P MidCap 400 / SmallCap 600 constituent pages.

Survivorship mitigation: besides today's constituents we add every ticker the index
*removed* since the start of the warm-up window (many of those still trade after
falling out of the index). Names that were delisted outright have no free price
history and remain a known bias.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests

from src.data.cache import DiskCache
from src.domain.models import UniverseMember

WIKI_PAGES = {
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
_HEADERS = {"User-Agent": "make-it-rain-research/0.1 (backtest universe builder)"}


def to_yahoo_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".", "-")


def _fetch_tables(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join(str(x) for x in dict.fromkeys(c)).strip() for c in df.columns]
    return df


def parse_constituents(tables: list[pd.DataFrame], index: str) -> list[UniverseMember]:
    current = _flatten(tables[0])
    sym_col = next(c for c in current.columns if c.lower() in ("symbol", "ticker symbol", "ticker"))
    name_col = next((c for c in current.columns if c.lower() in ("security", "company")), None)
    members = [
        UniverseMember(
            symbol=to_yahoo_symbol(str(r[sym_col])),
            name=str(r[name_col]) if name_col else None,
            source=f"wikipedia:{index}:current",
        )
        for _, r in current.iterrows()
        if isinstance(r[sym_col], str)
    ]
    return members


def parse_removals(tables: list[pd.DataFrame], index: str, since: date) -> list[UniverseMember]:
    out: list[UniverseMember] = []
    for raw in tables[1:]:
        df = _flatten(raw)
        date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
        rem_col = next((c for c in df.columns if c.lower().startswith("removed") and "ticker" in c.lower()), None)
        if date_col is None or rem_col is None:
            continue
        name_col = next((c for c in df.columns if c.lower().startswith("removed") and "security" in c.lower()), None)
        for _, r in df.iterrows():
            d = pd.to_datetime(str(r[date_col]).split("[")[0], errors="coerce")
            sym = r[rem_col]
            if pd.isna(d) or not isinstance(sym, str) or not sym.strip() or d.date() < since:
                continue
            out.append(
                UniverseMember(
                    symbol=to_yahoo_symbol(sym),
                    name=str(r[name_col]) if name_col and isinstance(r[name_col], str) else None,
                    source=f"wikipedia:{index}:removed",
                    removed_on=d.date(),
                )
            )
    return out


def load_index_universe(
    cache: DiskCache, indexes: list[str], include_removed_since: date | None
) -> list[UniverseMember]:
    members: dict[str, UniverseMember] = {}
    for index in indexes:
        url = WIKI_PAGES[index]
        cur_key, rem_key = f"{index}_current", f"{index}_removed"
        cur_df, rem_df = cache.read("universe", cur_key), cache.read("universe", rem_key)
        if cur_df is None or rem_df is None:
            tables = _fetch_tables(url)
            cur = parse_constituents(tables, index)
            rem = parse_removals(tables, index, date(1990, 1, 1))
            cur_df = pd.DataFrame([m.model_dump() for m in cur])
            rem_df = pd.DataFrame([m.model_dump() for m in rem])
            rem_df["removed_on"] = rem_df["removed_on"].astype(str)
            cache.write("universe", cur_key, cur_df)
            cache.write("universe", rem_key, rem_df)
        for rec in cur_df.to_dict("records"):
            members.setdefault(rec["symbol"], UniverseMember(**_clean(rec)))
        if include_removed_since is not None:
            for rec in rem_df.to_dict("records"):
                m = UniverseMember(**_clean(rec))
                if m.removed_on and m.removed_on >= include_removed_since:
                    members.setdefault(m.symbol, m)
    return sorted(members.values(), key=lambda m: m.symbol)


def _clean(rec: dict) -> dict:
    out = {k: (None if (isinstance(v, float) and pd.isna(v)) or v in ("None", "NaT") else v) for k, v in rec.items()}
    if out.get("removed_on"):
        out["removed_on"] = pd.to_datetime(out["removed_on"]).date()
    return out
