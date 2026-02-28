from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from stock_filter.core.utils import (
    normalize_universe_name,
    validate_yyyymmdd,
    today_yyyymmdd,
)
from stock_filter.datasource.cache import ParquetCache
from stock_filter.datasource.loader import DataLoader
from stock_filter.datasource.pykrx_client import PykrxClient
from stock_filter.datasource.ohlcv_series_store import OhlcvSeriesStore
from stock_filter.datasource.ohlcv_sync import OhlcvSync
from stock_filter.universe.krx_index import KrxIndexUniverseProvider
from stock_filter.universe.service import UniverseService
from stock_filter.universe.static_csv import StaticCsvUniverseProvider



def _normalize_markets(value: str) -> list[str]:
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p in ("kospi", "kspi"):
            out.append("KOSPI")
        elif p in ("kosdaq", "ksdq"):
            out.append("KOSDAQ")
        else:
            raise ValueError(f"Unsupported market: {p!r} (use: kospi, kosdaq)")
    # stable + unique
    return sorted(set(out))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stockfilter")
    sub = p.add_subparsers(dest="command", required=True)

    p_uni = sub.add_parser("universe", help="Print universe tickers (and names if available)")
    p_uni.add_argument("--universe", required=True, help="kospi100 | kosdaq100")
    p_uni.add_argument("--date", required=False, default=None, help="YYYYMMDD (defaults to today)")
    p_uni.add_argument("--no-names", action="store_true", help="Print only tickers (skip name lookups)")

    p_fetch = sub.add_parser("fetch", help="Fetch + cache OHLCV for a universe (window cache)")
    p_fetch.add_argument("--universe", required=True, help="kospi100 | kosdaq100")
    p_fetch.add_argument("--start", required=True, help="YYYYMMDD")
    p_fetch.add_argument("--end", required=True, help="YYYYMMDD")
    p_fetch.add_argument("--freq", required=False, default="d", choices=["d", "m", "y"], help="d|m|y")
    p_fetch.add_argument("--out", required=True, help="Output cache directory, e.g. data/cache")
    p_fetch.add_argument("--sleep", required=False, type=float, default=0.25, help="Sleep between requests (seconds)")
    p_fetch.add_argument("--max-retries", required=False, type=int, default=3, help="Max retries per ticker")

    # NEW: overwrite last N years for ALL tickers in markets (series store)
    p_backfill = sub.add_parser("backfill", help="Overwrite last N years OHLCV for all tickers in KOSPI/KOSDAQ")
    p_backfill.add_argument("--markets", required=False, default="kospi,kosdaq", help="Uses static kospi100/kosdaq100 CSVs. Choose: kospi, kosdaq, kospi,kosdaq")
    p_backfill.add_argument("--years", required=False, type=int, default=3, help="how many years to overwrite")
    p_backfill.add_argument("--freq", required=False, default="d", choices=["d", "m", "y"], help="d|m|y")
    p_backfill.add_argument("--out", required=True, help="Output cache directory, e.g. data/cache")
    p_backfill.add_argument("--asof", required=False, default=None, help="YYYYMMDD (defaults to today)")
    p_backfill.add_argument("--sleep", required=False, type=float, default=0.25)
    p_backfill.add_argument("--max-retries", required=False, type=int, default=3)
    p_backfill.add_argument("--limit", required=False, type=int, default=None, help="limit number of tickers (smoke test)")

    # NEW: append from last date to today (series store)
    p_update = sub.add_parser("update", help="Append new OHLCV rows up to today for all tickers in KOSPI/KOSDAQ")
    p_update.add_argument("--markets", required=False, default="kospi,kosdaq", help="Uses static kospi100/kosdaq100 CSVs. Choose: kospi, kosdaq, kospi,kosdaq")
    p_update.add_argument("--freq", required=False, default="d", choices=["d", "m", "y"], help="d|m|y")
    p_update.add_argument("--out", required=True, help="Output cache directory, e.g. data/cache")
    p_update.add_argument("--asof", required=False, default=None, help="YYYYMMDD (defaults to today)")
    p_update.add_argument("--bootstrap-years", required=False, type=int, default=3, help="if missing file, backfill this many years")
    p_update.add_argument("--sleep", required=False, type=float, default=0.25)
    p_update.add_argument("--max-retries", required=False, type=int, default=3)
    p_update.add_argument("--limit", required=False, type=int, default=None, help="limit number of tickers (smoke test)")

    return p


def cmd_universe(args: argparse.Namespace) -> int:
    universe = normalize_universe_name(args.universe)
    asof = args.date or today_yyyymmdd()
    validate_yyyymmdd(asof)

    client = PykrxClient()
    svc = UniverseService(
        primary=KrxIndexUniverseProvider(client=client),
        fallback=StaticCsvUniverseProvider(),
    )

    members = svc.get_members(universe=universe, asof=asof, with_names=not args.no_names)

    if args.no_names:
        for m in members:
            print(m.ticker)
    else:
        print("ticker,name")
        for m in members:
            print(f"{m.ticker},{m.name or ''}")

    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    universe = normalize_universe_name(args.universe)
    start = validate_yyyymmdd(args.start)
    end = validate_yyyymmdd(args.end)
    freq = args.freq

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = PykrxClient()
    svc = UniverseService(
        primary=KrxIndexUniverseProvider(client=client),
        fallback=StaticCsvUniverseProvider(),
    )
    members = svc.get_members(universe=universe, asof=end, with_names=False)
    tickers = [m.ticker for m in members]

    cache = ParquetCache(out_dir)
    loader = DataLoader(
        client=client,
        cache=cache,
        sleep_seconds=float(args.sleep),
        max_retries=int(args.max_retries),
    )

    t0 = time.perf_counter()
    stats = loader.fetch_many(tickers=tickers, start=start, end=end, freq=freq)
    elapsed = time.perf_counter() - t0

    print("")
    print("Fetch summary")
    print(f"  Universe: {universe} (asof={end})")
    print(f"  Tickers:  {stats.total}")
    print(f"  Cached:   {stats.from_cache}")
    print(f"  Fetched:  {stats.fetched}")
    print(f"  Failed:   {stats.failed}")
    print(f"  Elapsed:  {elapsed:.2f}s")
    if stats.failures:
        print("")
        print("Failures (ticker -> error):")
        for t, err in stats.failures[:20]:
            print(f"  {t} -> {err}")
        if len(stats.failures) > 20:
            print(f"  ... ({len(stats.failures) - 20} more)")

    return 0 if stats.failed == 0 else 2


def _load_static_100_tickers(markets_arg: str) -> list[str]:
    """
    Uses static CSVs:
      - src/stock_filter/universe/static/kospi100.csv
      - src/stock_filter/universe/static/kosdaq100.csv

    markets_arg: "kospi", "kosdaq", or "kospi,kosdaq"
    """
    parts = [p.strip().lower() for p in (markets_arg or "").split(",") if p.strip()]
    if not parts:
        parts = ["kospi", "kosdaq"]

    want_kospi = "kospi" in parts
    want_kosdaq = "kosdaq" in parts
    if not want_kospi and not want_kosdaq:
        raise ValueError("markets must include kospi and/or kosdaq (e.g., kospi,kosdaq)")

    provider = StaticCsvUniverseProvider()
    tickers: list[str] = []

    if want_kospi:
        members = provider.get_members(universe="kospi100", asof="00000000", with_names=False)
        tickers.extend([m.ticker for m in members])

    if want_kosdaq:
        members = provider.get_members(universe="kosdaq100", asof="00000000", with_names=False)
        tickers.extend([m.ticker for m in members])

    return sorted(set(tickers))


def cmd_backfill(args: argparse.Namespace) -> int:
    # NOTE: we do NOT use pykrx market ticker lists; we always use static kospi100/kosdaq100 CSVs
    tickers = _load_static_100_tickers(args.markets)

    freq = args.freq
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    asof = today_yyyymmdd() if args.asof is None else validate_yyyymmdd(args.asof)

    client = PykrxClient()
    store = OhlcvSeriesStore(out_dir)
    sync = OhlcvSync(
        client=client,
        store=store,
        sleep_seconds=float(args.sleep),
        max_retries=int(args.max_retries),
    )

    t0 = time.perf_counter()
    stats = sync.overwrite_last_years(
        tickers=tickers,
        years=int(args.years),
        freq=freq,
        asof=asof,
        limit=args.limit,
    )
    elapsed = time.perf_counter() - t0

    print("")
    print("Backfill summary (series store)")
    print(f"  Universe: static kospi100/kosdaq100 ({args.markets})")
    print(f"  As-of:    {asof}")
    print(f"  Years:    {args.years}")
    print(f"  Freq:     {freq}")
    print(f"  Tickers:  {stats.total}")
    print(f"  Written:  {stats.written}")
    print(f"  Failed:   {stats.failed}")
    print(f"  Elapsed:  {elapsed:.2f}s")
    if stats.failures:
        print("")
        print("Failures (ticker -> error):")
        for t, err in stats.failures[:20]:
            print(f"  {t} -> {err}")
        if len(stats.failures) > 20:
            print(f"  ... ({len(stats.failures) - 20} more)")
    return 0 if stats.failed == 0 else 2


def cmd_update(args: argparse.Namespace) -> int:
    # NOTE: we do NOT use pykrx market ticker lists; we always use static kospi100/kosdaq100 CSVs
    tickers = _load_static_100_tickers(args.markets)

    freq = args.freq
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    asof = today_yyyymmdd() if args.asof is None else validate_yyyymmdd(args.asof)

    client = PykrxClient()
    store = OhlcvSeriesStore(out_dir)
    sync = OhlcvSync(
        client=client,
        store=store,
        sleep_seconds=float(args.sleep),
        max_retries=int(args.max_retries),
    )

    t0 = time.perf_counter()
    stats = sync.update_to_today(
        tickers=tickers,
        freq=freq,
        asof=asof,
        bootstrap_years=int(args.bootstrap_years),
        limit=args.limit,
    )
    elapsed = time.perf_counter() - t0

    print("")
    print("Update summary (series store)")
    print(f"  Universe:      static kospi100/kosdaq100 ({args.markets})")
    print(f"  As-of:         {asof}")
    print(f"  Freq:          {freq}")
    print(f"  Tickers:       {stats.total}")
    print(f"  Updated:       {stats.updated}")
    print(f"  Bootstrapped:  {stats.bootstrapped}")
    print(f"  Skipped:       {stats.skipped}")
    print(f"  Failed:        {stats.failed}")
    print(f"  Elapsed:       {elapsed:.2f}s")
    if stats.failures:
        print("")
        print("Failures (ticker -> error):")
        for t, err in stats.failures[:20]:
            print(f"  {t} -> {err}")
        if len(stats.failures) > 20:
            print(f"  ... ({len(stats.failures) - 20} more)")
    return 0 if stats.failed == 0 else 2


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "universe":
        raise SystemExit(cmd_universe(args))
    if args.command == "fetch":
        raise SystemExit(cmd_fetch(args))
    if args.command == "backfill":
        raise SystemExit(cmd_backfill(args))
    if args.command == "update":
        raise SystemExit(cmd_update(args))

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])