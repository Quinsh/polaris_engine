# src/stock_filter/app/interactive.py

from __future__ import annotations

import sys
import time
from pathlib import Path

from stock_filter.analytics.pipeline import run_screen_pipeline
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
from stock_filter.risk import PositionSizingInput, calculate_position_size
from stock_filter.storage import load_series
from stock_filter.universe.krx_index import KrxIndexUniverseProvider
from stock_filter.universe.service import UniverseService
from stock_filter.universe.static_csv import StaticCsvUniverseProvider


TICKER_MARKET_MAP: dict[str, str] = {}
MARKET_INDEX_ETF = {
    "KOSPI": "226490",
    "KOSDAQ": "229200",
}


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


def average_signal_score(signal_scores: dict[str, float], selected_signals: list[str]) -> float:
    if not selected_signals:
        return 0.0
    values = [float(signal_scores.get(name, 0.0)) for name in selected_signals]
    return sum(values) / len(values)


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
        kospi_etf = MARKET_INDEX_ETF["KOSPI"]
        tickers.append(kospi_etf)
        TICKER_MARKET_MAP.setdefault(kospi_etf, "KOSPI")

    if want_kosdaq:
        members = provider.get_members(universe="kosdaq100", asof="00000000", with_names=False)
        for m in members:
            tickers.append(m.ticker)
            TICKER_MARKET_MAP.setdefault(m.ticker, "KOSDAQ")
        kosdaq_etf = MARKET_INDEX_ETF["KOSDAQ"]
        tickers.append(kosdaq_etf)
        TICKER_MARKET_MAP.setdefault(kosdaq_etf, "KOSDAQ")

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

    passed = sorted(
        passed,
        key=lambda r: average_signal_score(r.signal_scores, selected_signals),
        reverse=True,
    )

    print()
    print("Screen Summary")
    print(f"Considered: {len(tickers)}")
    print(f"Evaluated:  {len(results)}")
    print(f"Passed:     {len(passed)}")
    print(f"Skipped:    {skipped}")

    if passed:
        print()
        cols = ["ticker", "asof"]
        if selected_signals:
            cols.append("avg_score")
            cols.extend(selected_signals)
        print(",".join(cols))
        for row in passed:
            label = _fmt_ticker_with_market(row.ticker)
            if selected_signals:
                avg_score = average_signal_score(row.signal_scores, selected_signals)
                scores = [str(row.signal_scores.get(s, "")) for s in selected_signals]
                print(f"{label},{row.asof or ''},{avg_score}," + ",".join(scores))
            else:
                print(f"{label},{row.asof or ''}")
    print()



def run_position_size() -> None:
    print()

    supported_fields = set(getattr(PositionSizingInput, "__dataclass_fields__", {}).keys())

    def _optional_float(raw: str) -> float | None:
        value = (raw or "").strip()
        return float(value) if value else None

    ticker = prompt("Ticker (e.g. 005930)")
    account_size = float(prompt("Account size (KRW)") or "0")
    risk_percent = float(prompt("Risk percent per trade (e.g. 1 for 1%)") or "1") / 100.0
    entry_price = float(prompt("Entry price"))

    side_raw = (prompt("Side (long/short) [default=long]") or "long").strip().lower()
    if side_raw in {"l", "long"}:
        side = "long"
    elif side_raw in {"s", "short"}:
        side = "short"
    else:
        raise ValueError("side must be 'long' or 'short'")

    mode_raw = (prompt("Mode: fixed_stop / percent_stop / atr [default=atr]") or "atr").strip().lower()
    if mode_raw in {"fixed", "fixed stop", "fixed_stop"}:
        mode = "fixed_stop"
    elif mode_raw in {"percent", "percent stop", "percent_stop", "pct", "%"}:
        mode = "percent_stop"
    elif mode_raw in {"atr", "a"}:
        mode = "atr"
    else:
        raise ValueError("mode must be fixed_stop, percent_stop, or atr")

    params: dict[str, object] = {
        "ticker": ticker,
        "account_size": account_size,
        "risk_percent": risk_percent,
        "entry_price": entry_price,
    }

    explain_raw = (prompt("Show colorful step-by-step derivation? (Y/n)") or "y").strip().lower()
    explain_steps = explain_raw not in {"n", "no"}
    delay_seconds = float(prompt("Delay per derivation step in seconds [default=0.35]") or "0.35")
    if "explain_steps" in supported_fields:
        params["explain_steps"] = explain_steps
    if "step_delay_seconds" in supported_fields:
        params["step_delay_seconds"] = delay_seconds

    if "side" in supported_fields:
        params["side"] = side
    if "sizing_method" in supported_fields:
        params["sizing_method"] = mode

    advanced_raw = (prompt("Use advanced controls? (y/N)") or "n").strip().lower()
    use_advanced = advanced_raw in {"y", "yes"}

    if use_advanced:
        available_capital = _optional_float(
            prompt("Available capital / buying power (KRW) [blank=account_size × leverage]")
        )
        max_position_percent_raw = prompt(
            "Max position as % of account [blank=no extra cap]"
        )
        max_leverage = float(prompt("Max leverage [default=1]") or "1")
        lot_size = int(prompt("Lot size [default=1]") or "1")
        minimum_quantity = int(prompt("Minimum quantity [default=0]") or "0")
        commission_per_share = float(prompt("Commission per share [default=0]") or "0")
        slippage_per_share = float(prompt("Slippage per share [default=0]") or "0")
        fixed_fees = float(prompt("Fixed fees per trade [default=0]") or "0")

        if "available_capital" in supported_fields and available_capital is not None:
            params["available_capital"] = available_capital
        if "max_position_percent" in supported_fields and max_position_percent_raw.strip():
            params["max_position_percent"] = float(max_position_percent_raw) / 100.0
        if "max_leverage" in supported_fields:
            params["max_leverage"] = max_leverage
        if "lot_size" in supported_fields:
            params["lot_size"] = lot_size
        if "minimum_quantity" in supported_fields:
            params["minimum_quantity"] = minimum_quantity
        if "commission_per_share" in supported_fields:
            params["commission_per_share"] = commission_per_share
        if "slippage_per_share" in supported_fields:
            params["slippage_per_share"] = slippage_per_share
        if "fixed_fees" in supported_fields:
            params["fixed_fees"] = fixed_fees

    if mode == "fixed_stop":
        stop_price = float(prompt("Stop price"))
        params["stop_price"] = stop_price

    elif mode == "percent_stop":
        stop_percent = float(prompt("Stop percent (e.g. 5 for 5%)") or "0") / 100.0
        if "stop_percent" in supported_fields:
            params["stop_percent"] = stop_percent
        else:
            stop_distance = entry_price * stop_percent
            stop_price = entry_price - stop_distance if side == "long" else entry_price + stop_distance
            params["stop_price"] = stop_price

    else:
        out_dir = Path(prompt("Cache directory [default=data/cache]") or "data/cache")
        freq = prompt("Frequency (d/m/y) [default=d]") or "d"
        atr_period = int(prompt("ATR period [default=14]") or "14")
        atr_multiplier = float(prompt("ATR multiplier [default=2]") or "2")
        atr_method = (prompt("ATR method (wilder/sma) [default=wilder]") or "wilder").strip().lower()

        try:
            df = load_series(ticker=ticker, freq=freq, root_dir=out_dir)
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            print("Backfill OHLCV series first (option 3 or 4).\n")
            return

        params["ohlcv"] = df
        if "atr_period" in supported_fields:
            params["atr_period"] = atr_period
        if "atr_multiplier" in supported_fields:
            params["atr_multiplier"] = atr_multiplier
        if "atr_method" in supported_fields:
            params["atr_method"] = atr_method

    result = calculate_position_size(PositionSizingInput(**params))

    print()
    print("Position Size Summary")
    print(f"Ticker:               {getattr(result, 'ticker', ticker)}")
    if hasattr(result, "side"):
        print(f"Side:                 {result.side}")
    print(f"Method:               {result.method}")
    if hasattr(result, "entry_price"):
        print(f"Entry price:          {result.entry_price:,.2f}")
    if hasattr(result, "stop_price"):
        print(f"Stop price:           {result.stop_price:,.2f}")
    if hasattr(result, "stop_distance"):
        print(f"Stop distance:        {result.stop_distance:,.2f}")
    if getattr(result, "atr_value", None) is not None:
        print(f"ATR:                  {result.atr_value:,.4f}")
    print(f"Risk amount:          {result.risk_amount:,.2f}")
    print(f"Risk/share:           {result.risk_per_share:,.4f}")
    print(f"Quantity:             {result.quantity:,}")
    print(f"Position value:       {result.position_value:,.2f}")
    if hasattr(result, "quantity_by_risk"):
        print(f"Qty allowed by risk:  {result.quantity_by_risk:,}")
    if hasattr(result, "quantity_by_capital"):
        print(f"Qty allowed by cap:   {result.quantity_by_capital:,}")
    if hasattr(result, "max_position_value"):
        print(f"Max position value:   {result.max_position_value:,.2f}")
    if hasattr(result, "actual_risk_amount"):
        print(f"Actual risk amount:   {result.actual_risk_amount:,.2f}")
    if hasattr(result, "actual_risk_percent"):
        print(f"Actual risk %:        {result.actual_risk_percent * 100:.3f}%")
    if hasattr(result, "binding_constraint"):
        print(f"Binding constraint:   {result.binding_constraint}")
    notes = getattr(result, "notes", ())
    if notes:
        print("Notes:")
        for note in notes:
            print(f"  - {note}")
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
        print("6) Calculate Position Size")
        print("7) Exit")
        print()

        choice = input("Enter choice (1-7): ").strip()

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
                run_position_size()
            elif choice == "7":
                print("Exiting.")
                sys.exit(0)
            else:
                print("Invalid selection.\n")
        except KeyboardInterrupt:
            print("\nInterrupted.\n")
        except Exception as e:  # noqa: BLE001
            print(f"\nError: {e}\n")
