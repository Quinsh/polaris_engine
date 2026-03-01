# src/stock_filter/app/interactive.py

from __future__ import annotations

import sys
import time
from pathlib import Path

from stock_filter.analytics.pipeline import run_screen_pipeline
from stock_filter.analytics.types import ScreenResult
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
from stock_filter.features import FEATURE_REGISTRY
from stock_filter.screening import build_feature_rules, build_signal_rules
from stock_filter.signals import SIGNAL_REGISTRY
from stock_filter.universe.krx_index import KrxIndexUniverseProvider
from stock_filter.universe.service import UniverseService
from stock_filter.universe.static_csv import StaticCsvUniverseProvider


TICKER_MARKET_MAP: dict[str, str] = {}


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


def _parse_space_separated(
    text: str,
    available: list[str],
    kind: str,
    default_all: bool = True,
) -> list[str]:
    """Parse space-separated input; validate against available; return subset or all if default_all."""
    chosen = [x.strip() for x in (text or "").split() if x.strip()]
    if not chosen:
        return list(available) if default_all else []
    valid = [c for c in chosen if c in available]
    invalid = [c for c in chosen if c not in available]
    if invalid:
        print(f"[WARN] Unknown {kind}: {invalid}; using: {valid or available}")
    return valid if valid else (list(available) if default_all else [])


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
        for m in members:
            tickers.append(m.ticker)
            TICKER_MARKET_MAP.setdefault(m.ticker, "KOSPI")

    if want_kosdaq:
        members = provider.get_members(universe="kosdaq100", asof="00000000", with_names=False)
        for m in members:
            tickers.append(m.ticker)
            TICKER_MARKET_MAP.setdefault(m.ticker, "KOSDAQ")

    return sorted(set(tickers))


def _fmt_ticker_with_market(ticker: str) -> str:
    market = TICKER_MARKET_MAP.get(ticker, "UNKNOWN")
    return f"{ticker}({market})"


def _progress_line(i: int, total: int, ok: int, skipped: int, failed: int, ticker: str, msg: str = "") -> None:
    # Single-line updating status
    line = f"[{i:>4}/{total:<4}] ok={ok:<4} skip={skipped:<4} fail={failed:<4}  {_fmt_ticker_with_market(ticker)} {msg}"
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
    market_label = "KOSPI" if universe == "kospi100" else "KOSDAQ"
    for t in tickers:
        TICKER_MARKET_MAP.setdefault(t, market_label)

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
    market_label = "KOSPI" if universe == "kospi100" else "KOSDAQ"

    print()
    print(f"{universe.upper()} members ({len(members)})")
    print("-" * 40)
    for m in members:
        TICKER_MARKET_MAP.setdefault(m.ticker, market_label)
        print(f"{_fmt_ticker_with_market(m.ticker)}  {m.name or ''}")
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


def run_screen() -> None:
    print()
    available_features = sorted(FEATURE_REGISTRY.keys())
    available_signals = sorted(SIGNAL_REGISTRY.keys())
    print("Available features:", " ".join(available_features))
    print("Available signals: ", " ".join(available_signals))
    print()

    features_raw = prompt("Features (space-separated) [default=none]")
    signals_raw = prompt("Signals (space-separated) [default=none]")
    selected_features = _parse_space_separated(
        features_raw, available_features, "features", default_all=False
    )
    selected_signals = _parse_space_separated(
        signals_raw, available_signals, "signals", default_all=False
    )

    feature_rules = build_feature_rules(selected_features)
    signal_rules = build_signal_rules(selected_signals)
    if not feature_rules and not signal_rules:
        print(
            "[WARN] No default rules for selected items; all tickers will pass. "
            "Add entries in screening/rule_defaults.py for filtering."
        )

    ticker = prompt("Single ticker (e.g. 005930) [leave blank for markets]")
    markets_raw = ""
    if not ticker.strip():
        markets_raw = prompt("Which static lists? (kospi,kosdaq) [default=kospi,kosdaq]") or "kospi,kosdaq"
    freq = prompt("Frequency (d/m/y) [default=d]") or "d"
    out_dir = Path(prompt("Cache directory [default=data/cache]") or "data/cache")
    limit_raw = prompt("Limit tickers (smoke test) [default=none]") or ""

    limit = int(limit_raw) if limit_raw.strip() else None

    if ticker.strip():
        tickers = [ticker.strip()]
    else:
        tickers = _load_static_100_tickers(markets_raw)

    if limit is not None:
        tickers = tickers[: max(0, limit)]

    if not tickers:
        print("No tickers to screen.\n")
        return

    config = {
        "features": selected_features,
        "signals": selected_signals,
        "rules": {"feature_rules": feature_rules, "signal_rules": signal_rules},
    }

    print()
    print(f"Screening {len(tickers)} ticker(s) with features={selected_features} signals={selected_signals}...")

    results = []
    skipped = 0
    for tkr in tickers:
        try:
            result = run_screen_pipeline(ticker=tkr, freq=freq, root_dir=out_dir, config=config)
            results.append(result)
        except FileNotFoundError as exc:
            skipped += 1
            print(f"[WARN] skipping {_fmt_ticker_with_market(tkr)}: {exc}")

    passed = [r for r in results if r.passed]

    def _sort_key(r: ScreenResult) -> float:
        if selected_signals:
            return r.signal_scores.get(selected_signals[0], 0.0)
        return 0.0

    passed = sorted(passed, key=_sort_key, reverse=True)

    print()
    print("Screen Summary")
    print(f"Considered: {len(tickers)}")
    print(f"Evaluated:  {len(results)}")
    print(f"Passed:     {len(passed)}")
    print(f"Skipped:    {skipped}")

    if passed:
        print()
        cols = ["ticker", "asof"] + selected_signals
        print(",".join(cols))
        for row in passed:
            label = _fmt_ticker_with_market(row.ticker)
            if selected_signals:
                scores = [str(row.signal_scores.get(s, "")) for s in selected_signals]
                print(f"{label},{row.asof or ''}," + ",".join(scores))
            else:
                print(f"{label},{row.asof or ''}")
    print()


def main() -> None:
    print_banner()

    while True:
        print("Select an option:")
        print("1) Show Universe")
        print("2) Fetch OHLCV Data (window cache)")
        print("3) Backfill OHLCV Series (static kospi100/kosdaq100)")
        print("4) Update OHLCV Series to Today (static kospi100/kosdaq100)")
        print("5) Screen cached series")
        print("6) Exit")
        print()

        choice = input("Enter choice (1-6): ").strip()

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
                run_screen()
            elif choice == "6":
                print("Exiting.")
                sys.exit(0)
            else:
                print("Invalid selection.\n")
        except KeyboardInterrupt:
            print("\nInterrupted.\n")
        except Exception as e:  # noqa: BLE001
            print(f"\nError: {e}\n")