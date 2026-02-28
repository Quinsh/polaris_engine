# src/stock_filter/app/interactive.py

from __future__ import annotations

import sys
import time
from pathlib import Path

from stock_filter.core.utils import (
    normalize_universe_name,
    today_yyyymmdd,
    validate_yyyymmdd,
)
from stock_filter.datasource.cache import ParquetCache
from stock_filter.datasource.loader import DataLoader
from stock_filter.datasource.ohlcv_series_store import OhlcvSeriesStore
from stock_filter.datasource.ohlcv_sync import OhlcvSync
from stock_filter.datasource.pykrx_client import PykrxClient
from stock_filter.universe.krx_index import KrxIndexUniverseProvider
from stock_filter.universe.service import UniverseService
from stock_filter.universe.static_csv import StaticCsvUniverseProvider


def print_banner() -> None:
    print("=" * 60)
    print(" KOREAN STOCK FILTER ENGINE (Prototype)")
    print("=" * 60)
    print()


def prompt(text: str) -> str:
    return input(f"{text}: ").strip()


def _parse_markets(markets_raw: str) -> tuple[bool, bool]:
    """
    Returns (want_kospi, want_kosdaq) for inputs like:
      - "kospi"
      - "kosdaq"
      - "kospi,kosdaq"
      - "" -> default both
    """
    s = (markets_raw or "").strip().lower()
    if not s:
        return True, True
    parts = [p.strip() for p in s.split(",") if p.strip()]
    want_kospi = "kospi" in parts
    want_kosdaq = "kosdaq" in parts
    if not want_kospi and not want_kosdaq:
        raise ValueError("markets must include kospi and/or kosdaq (e.g., kospi,kosdaq)")
    return want_kospi, want_kosdaq


def _load_static_100_tickers(markets_raw: str) -> list[str]:
    """
    Loads tickers from static universe CSVs:
      - kospi100.csv
      - kosdaq100.csv
    """
    want_kospi, want_kosdaq = _parse_markets(markets_raw)

    provider = StaticCsvUniverseProvider()
    tickers: list[str] = []

    if want_kospi:
        members = provider.get_members(universe="kospi100", asof="00000000", with_names=False)
        tickers.extend([m.ticker for m in members])

    if want_kosdaq:
        members = provider.get_members(universe="kosdaq100", asof="00000000", with_names=False)
        tickers.extend([m.ticker for m in members])

    return sorted(set(tickers))


def _progress_line(i: int, total: int, ok: int, skipped: int, failed: int, ticker: str, msg: str = "") -> None:
    # Single-line updating status
    line = f"[{i:>4}/{total:<4}] ok={ok:<4} skip={skipped:<4} fail={failed:<4}  {ticker} {msg}"
    print("\r" + line.ljust(90), end="", flush=True)


def run_fetch() -> None:
    universe = normalize_universe_name(prompt("Universe (kospi100 / kosdaq100)"))
    start = validate_yyyymmdd(prompt("Start date (YYYYMMDD)"))
    end = validate_yyyymmdd(prompt("End date (YYYYMMDD)"))
    freq = prompt("Frequency (d/m/y) [default=d]") or "d"
    out_dir = Path(prompt("Cache directory [default=data/cache]") or "data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    client = PykrxClient()
    svc = UniverseService(
        primary=KrxIndexUniverseProvider(client=client),
        fallback=StaticCsvUniverseProvider(),
    )

    members = svc.get_members(universe=universe, asof=end, with_names=False)
    tickers = [m.ticker for m in members]

    print()
    print(f"Universe resolved: {len(tickers)} tickers")
    print("Starting fetch...")
    print()

    cache = ParquetCache(out_dir)
    loader = DataLoader(client=client, cache=cache)

    t0 = time.perf_counter()
    stats = loader.fetch_many(tickers=tickers, start=start, end=end, freq=freq)
    elapsed = time.perf_counter() - t0

    print()
    print("Fetch Summary")
    print(f"Total:    {stats.total}")
    print(f"Fetched:  {stats.fetched}")
    print(f"Cached:   {stats.from_cache}")
    print(f"Failed:   {stats.failed}")
    print(f"Elapsed:  {elapsed:.2f}s")
    print()


def run_universe() -> None:
    universe = normalize_universe_name(prompt("Universe (kospi100 / kosdaq100)"))
    date = validate_yyyymmdd(prompt("As-of date (YYYYMMDD)"))

    client = PykrxClient()
    svc = UniverseService(
        primary=KrxIndexUniverseProvider(client=client),
        fallback=StaticCsvUniverseProvider(),
    )

    members = svc.get_members(universe=universe, asof=date, with_names=True)

    print()
    print(f"{universe.upper()} members ({len(members)})")
    print("-" * 40)
    for m in members:
        print(f"{m.ticker}  {m.name or ''}")
    print()


def run_backfill_series() -> None:
    print()
    markets_raw = prompt("Which static lists? (kospi,kosdaq) [default=kospi,kosdaq]") or "kospi,kosdaq"
    years_raw = prompt("Backfill years [default=3]") or "3"
    freq = prompt("Frequency (d/m/y) [default=d]") or "d"
    out_dir = Path(prompt("Cache directory [default=data/cache]") or "data/cache")
    asof = prompt("As-of date YYYYMMDD [default=today]") or today_yyyymmdd()
    limit_raw = prompt("Limit tickers (smoke test) [default=none]") or ""
    sleep_raw = prompt("Sleep between requests seconds [default=0.25]") or "0.25"
    retries_raw = prompt("Max retries [default=3]") or "3"

    years = int(years_raw)
    validate_yyyymmdd(asof)
    limit = int(limit_raw) if limit_raw.strip() else None
    sleep_s = float(sleep_raw)
    max_retries = int(retries_raw)

    tickers = _load_static_100_tickers(markets_raw)
    if limit is not None:
        tickers = tickers[: max(0, limit)]

    print()
    print(f"Backfill (series) tickers={len(tickers)} years={years} asof={asof} freq={freq}")
    print("Starting...")

    out_dir.mkdir(parents=True, exist_ok=True)

    client = PykrxClient()
    store = OhlcvSeriesStore(out_dir)
    sync = OhlcvSync(client=client, store=store, sleep_seconds=sleep_s, max_retries=max_retries)

    # Visual progress: per ticker overwrite
    t0 = time.perf_counter()
    ok = skipped = failed = 0

    # We run per-ticker by calling sync._fetch... indirectly would require API changes.
    # Minimal approach: call overwrite_last_years with limit=None isn't granular.
    # So we call overwrite_last_years once per ticker to show progress.
    # (Still uses your sync logic and store.)
    for i, tkr in enumerate(tickers, start=1):
        try:
            # overwrite_last_years expects a list; pass [tkr]
            stats = sync.overwrite_last_years(tickers=[tkr], years=years, freq=freq, asof=asof, limit=None)
            if stats.failed == 0 and stats.written == 1:
                ok += 1
                _progress_line(i, len(tickers), ok, skipped, failed, tkr, "OK")
            else:
                # Should be rare; treat as failed if not written
                failed += 1
                _progress_line(i, len(tickers), ok, skipped, failed, tkr, "FAILED")
        except Exception as e:  # noqa: BLE001
            failed += 1
            _progress_line(i, len(tickers), ok, skipped, failed, tkr, f"ERR {type(e).__name__}")
            continue

    elapsed = time.perf_counter() - t0
    print()
    print()
    print("Backfill Summary (series)")
    print(f"Tickers:  {len(tickers)}")
    print(f"OK:       {ok}")
    print(f"Skipped:  {skipped}")
    print(f"Failed:   {failed}")
    print(f"Elapsed:  {elapsed:.2f}s")
    print()


def run_update_series() -> None:
    print()
    markets_raw = prompt("Which static lists? (kospi,kosdaq) [default=kospi,kosdaq]") or "kospi,kosdaq"
    freq = prompt("Frequency (d/m/y) [default=d]") or "d"
    out_dir = Path(prompt("Cache directory [default=data/cache]") or "data/cache")
    asof = prompt("As-of date YYYYMMDD [default=today]") or today_yyyymmdd()
    bootstrap_raw = prompt("Bootstrap years if missing [default=3]") or "3"
    limit_raw = prompt("Limit tickers (smoke test) [default=none]") or ""
    sleep_raw = prompt("Sleep between requests seconds [default=0.25]") or "0.25"
    retries_raw = prompt("Max retries [default=3]") or "3"

    validate_yyyymmdd(asof)
    bootstrap_years = int(bootstrap_raw)
    limit = int(limit_raw) if limit_raw.strip() else None
    sleep_s = float(sleep_raw)
    max_retries = int(retries_raw)

    tickers = _load_static_100_tickers(markets_raw)
    if limit is not None:
        tickers = tickers[: max(0, limit)]

    print()
    print(f"Update (series) tickers={len(tickers)} asof={asof} freq={freq} bootstrap={bootstrap_years}")
    print("Starting...")

    out_dir.mkdir(parents=True, exist_ok=True)

    client = PykrxClient()
    store = OhlcvSeriesStore(out_dir)
    sync = OhlcvSync(client=client, store=store, sleep_seconds=sleep_s, max_retries=max_retries)

    t0 = time.perf_counter()
    ok = skipped = failed = 0

    for i, tkr in enumerate(tickers, start=1):
        try:
            stats = sync.update_to_today(
                tickers=[tkr],
                freq=freq,
                asof=asof,
                bootstrap_years=bootstrap_years,
                limit=None,
            )

            if stats.failed > 0:
                failed += 1
                _progress_line(i, len(tickers), ok, skipped, failed, tkr, "FAILED")
            else:
                # If it updated or skipped is determined by internal logic; reflect it
                if stats.updated > 0 or stats.bootstrapped > 0:
                    ok += 1
                    _progress_line(i, len(tickers), ok, skipped, failed, tkr, "OK")
                else:
                    skipped += 1
                    _progress_line(i, len(tickers), ok, skipped, failed, tkr, "SKIP")
        except Exception as e:  # noqa: BLE001
            failed += 1
            _progress_line(i, len(tickers), ok, skipped, failed, tkr, f"ERR {type(e).__name__}")
            continue

    elapsed = time.perf_counter() - t0
    print()
    print()
    print("Update Summary (series)")
    print(f"Tickers:  {len(tickers)}")
    print(f"OK:       {ok}")
    print(f"Skipped:  {skipped}")
    print(f"Failed:   {failed}")
    print(f"Elapsed:  {elapsed:.2f}s")
    print()


def main() -> None:
    print_banner()

    while True:
        print("Select an option:")
        print("1) Show Universe")
        print("2) Fetch OHLCV Data (window cache)")
        print("3) Backfill OHLCV Series (static kospi100/kosdaq100)")
        print("4) Update OHLCV Series to Today (static kospi100/kosdaq100)")
        print("5) Exit")
        print()

        choice = input("Enter choice (1-5): ").strip()

        try:
            if choice == "1":
                run_universe()
            elif choice == "2":
                run_fetch()
            elif choice == "3":
                run_backfill_series()
            elif choice == "4":
                run_update_series()
            elif choice == "5":
                print("Exiting.")
                sys.exit(0)
            else:
                print("Invalid selection.\n")
        except KeyboardInterrupt:
            print("\nInterrupted.\n")
        except Exception as e:  # noqa: BLE001
            print(f"\nError: {e}\n")