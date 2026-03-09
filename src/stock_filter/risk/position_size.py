"""Risk-aware position sizing engine.

This module preserves the original public API while expanding the sizing logic
into a fuller, production-oriented engine.

Main features
-------------
- Fixed-stop sizing, percent-stop sizing, and ATR-based sizing
- Long/short-aware stop validation and ATR stop placement
- Capital / buying-power caps
- Lot-size rounding
- Slippage / commission / fixed-fee aware risk sizing
- Rich diagnostics so callers can see why a size was reduced
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["long", "short"]
SizingMethod = Literal["auto", "fixed_stop", "percent_stop", "atr"]
ResolvedSizingMethod = Literal["fixed_stop", "percent_stop", "atr"]
ATRMethod = Literal["wilder", "sma"]
BindingConstraint = Literal[
    "risk",
    "capital",
    "risk_and_capital",
    "minimum_quantity",
    "no_trade",
]


def compute_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Return the True Range series."""
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
    method: ATRMethod = "wilder",
) -> pd.Series:
    """Average True Range over ``n`` bars."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if method not in {"wilder", "sma"}:
        raise ValueError("method must be 'wilder' or 'sma'")

    tr = compute_true_range(high, low, close)

    if method == "sma":
        return tr.rolling(n, min_periods=n).mean()

    # Exact Wilder ATR: seed with the first n-period TR average, then recurse.
    atr = pd.Series(np.nan, index=tr.index, dtype="float64")
    valid_tr = tr.dropna()
    if len(valid_tr) < n:
        return atr

    first_valid_idx = valid_tr.index[n - 1]
    seed = float(valid_tr.iloc[:n].mean())
    atr.loc[first_valid_idx] = seed

    start_pos = tr.index.get_loc(first_valid_idx)
    prev_atr = seed
    for pos in range(start_pos + 1, len(tr)):
        tr_value = tr.iloc[pos]
        if pd.isna(tr_value):
            atr.iloc[pos] = np.nan
            continue
        prev_atr = ((prev_atr * (n - 1)) + float(tr_value)) / n
        atr.iloc[pos] = prev_atr

    return atr


@dataclass(frozen=True)
class PositionSizingInput:
    """Inputs for position sizing.

    Notes
    -----
    - ``risk_percent`` is interpreted as a fraction of account equity.
      For example, use ``0.01`` for 1% risk.
    - If ``sizing_method='auto'``, priority is:
      ``stop_price`` -> ``stop_percent`` -> ``ohlcv``.
    - ``available_capital`` defaults to ``account_size * max_leverage``.
    - ``max_position_percent`` caps notional exposure as a fraction of
      ``account_size``.
    - ``commission_per_share`` and ``slippage_per_share`` are treated as
      *per-side* frictions and are doubled internally to estimate round-trip risk.
    """

    ticker: str
    account_size: float
    risk_percent: float
    entry_price: float

    side: Side = "long"
    sizing_method: SizingMethod = "auto"

    # Explicit stop modes
    stop_price: float | None = None
    stop_percent: float | None = None

    # ATR mode
    ohlcv: pd.DataFrame | None = None
    atr_period: int = 14
    atr_multiplier: float = 2.0
    atr_method: ATRMethod = "wilder"

    # Portfolio / execution controls
    available_capital: float | None = None
    max_position_percent: float | None = None
    max_leverage: float = 1.0
    lot_size: int = 1
    minimum_quantity: int = 0

    # Trading frictions
    commission_per_share: float = 0.0
    slippage_per_share: float = 0.0
    fixed_fees: float = 0.0

    # Interactive explanation output controls
    explain_steps: bool = False
    step_delay_seconds: float = 0.0


@dataclass(frozen=True)
class PositionSizingResult:
    ticker: str
    risk_amount: float
    risk_per_share: float
    quantity: int
    position_value: float
    method: ResolvedSizingMethod
    atr_value: float | None = None

    side: Side = "long"
    entry_price: float = 0.0
    stop_price: float = 0.0
    stop_distance: float = 0.0
    quantity_by_risk: int = 0
    quantity_by_capital: int = 0
    raw_quantity: int = 0
    max_position_value: float = 0.0
    actual_risk_amount: float = 0.0
    actual_risk_percent: float = 0.0
    binding_constraint: BindingConstraint = "no_trade"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def actual_risk(self) -> float:
        return self.actual_risk_amount




class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"


def _emit_step(enabled: bool, title: str, detail: str, *, delay_seconds: float = 0.0) -> None:
    if not enabled:
        return
    print(f"{_Color.BOLD}{_Color.CYAN}[Position Sizing] {title}{_Color.RESET}")
    print(f"{_Color.GREEN}{detail}{_Color.RESET}")
    if delay_seconds > 0:
        time.sleep(delay_seconds)

def _round_amount(value: float, ndigits: int = 2) -> float:
    return round(float(value), ndigits)


def _round_price(value: float, ndigits: int = 4) -> float:
    return round(float(value), ndigits)


def _validate_fraction(value: float, *, field_name: str) -> None:
    if not (0 < value <= 1):
        raise ValueError(f"{field_name} must be in (0, 1]")


def _validate_positive(value: float, *, field_name: str, allow_zero: bool = False) -> None:
    ok = value >= 0 if allow_zero else value > 0
    if not ok:
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{field_name} must be {comparator}")


def _normalize_column_name(name: object) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _prepare_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    if ohlcv.empty:
        raise ValueError("ohlcv must not be empty")

    normalized_lookup = {_normalize_column_name(col): col for col in ohlcv.columns}

    required_aliases = {
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "adjclose", "c"),
    }

    selected: dict[str, pd.Series] = {}
    for canonical, aliases in required_aliases.items():
        for alias in aliases:
            if alias in normalized_lookup:
                selected[canonical] = pd.to_numeric(
                    ohlcv[normalized_lookup[alias]],
                    errors="coerce",
                )
                break
        else:
            raise ValueError(
                "ohlcv must contain high, low, and close columns "
                "(case-insensitive; 'Adj Close' is also accepted for close)"
            )

    out = pd.DataFrame(selected).dropna(subset=["high", "low", "close"])
    try:
        out = out.sort_index()
    except Exception:
        pass

    if out.empty:
        raise ValueError("ohlcv has no valid high/low/close rows")
    if (out["high"] <= 0).any() or (out["low"] <= 0).any() or (out["close"] <= 0).any():
        raise ValueError("ohlcv prices must be > 0")
    if (out["high"] < out["low"]).any():
        raise ValueError("ohlcv contains rows where high < low")
    return out


def _resolve_sizing_method(params: PositionSizingInput) -> ResolvedSizingMethod:
    if params.sizing_method == "fixed_stop":
        if params.stop_price is None:
            raise ValueError("sizing_method='fixed_stop' requires stop_price")
        return "fixed_stop"

    if params.sizing_method == "percent_stop":
        if params.stop_percent is None:
            raise ValueError("sizing_method='percent_stop' requires stop_percent")
        return "percent_stop"

    if params.sizing_method == "atr":
        if params.ohlcv is None or params.ohlcv.empty:
            raise ValueError("sizing_method='atr' requires non-empty ohlcv")
        return "atr"

    # auto mode: explicit stop takes priority, then percent stop, then ATR.
    if params.stop_price is not None:
        return "fixed_stop"
    if params.stop_percent is not None:
        return "percent_stop"
    if params.ohlcv is not None and not params.ohlcv.empty:
        return "atr"
    raise ValueError("Provide stop_price, stop_percent, or ohlcv for position sizing")


def _floor_to_lot(quantity: int, lot_size: int) -> int:
    if quantity <= 0:
        return 0
    return (quantity // lot_size) * lot_size


def _determine_stop_and_risk(
    params: PositionSizingInput,
    method: ResolvedSizingMethod,
) -> tuple[float, float, float | None, tuple[str, ...]]:
    notes: list[str] = []
    atr_value: float | None = None

    if method == "fixed_stop":
        assert params.stop_price is not None
        stop_price = float(params.stop_price)
        _validate_positive(stop_price, field_name="stop_price")
        stop_distance = abs(params.entry_price - stop_price)
        if stop_distance == 0:
            raise ValueError("entry_price and stop_price must be different")

        if params.side == "long" and stop_price >= params.entry_price:
            raise ValueError("For a long position, stop_price must be below entry_price")
        if params.side == "short" and stop_price <= params.entry_price:
            raise ValueError("For a short position, stop_price must be above entry_price")

        return stop_price, stop_distance, atr_value, tuple(notes)

    if method == "percent_stop":
        assert params.stop_percent is not None
        _validate_fraction(params.stop_percent, field_name="stop_percent")
        if params.stop_percent >= 1:
            raise ValueError("stop_percent must be < 1")
        stop_distance = params.entry_price * params.stop_percent
        stop_price = (
            params.entry_price - stop_distance
            if params.side == "long"
            else params.entry_price + stop_distance
        )
        return float(stop_price), float(stop_distance), atr_value, tuple(notes)

    ohlcv = _prepare_ohlcv(params.ohlcv if params.ohlcv is not None else pd.DataFrame())
    atr_series = compute_atr(
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        n=params.atr_period,
        method=params.atr_method,
    ).dropna()

    if atr_series.empty:
        raise ValueError(
            f"ATR could not be computed for {params.ticker} "
            f"(need at least {params.atr_period} valid bars)"
        )

    atr_value = float(atr_series.iloc[-1])
    if not (pd.notna(atr_value) and atr_value > 0):
        raise ValueError("computed ATR must be a positive finite number")

    stop_distance = atr_value * params.atr_multiplier
    if stop_distance <= 0:
        raise ValueError("ATR-based stop distance must be > 0")

    stop_price = (
        params.entry_price - stop_distance
        if params.side == "long"
        else params.entry_price + stop_distance
    )
    return float(stop_price), float(stop_distance), atr_value, tuple(notes)


def _binding_constraint(
    quantity: int,
    *,
    quantity_by_risk: int,
    quantity_by_capital: int,
    minimum_quantity: int,
) -> BindingConstraint:
    if quantity == 0:
        if minimum_quantity > 0:
            return "minimum_quantity"
        return "no_trade"
    if quantity_by_risk == quantity_by_capital == quantity:
        return "risk_and_capital"
    if quantity == quantity_by_risk and quantity < quantity_by_capital:
        return "risk"
    if quantity == quantity_by_capital and quantity < quantity_by_risk:
        return "capital"
    return "risk_and_capital"


def calculate_position_size(params: PositionSizingInput) -> PositionSizingResult:
    """Calculate a position size using a fixed stop, percent stop, or ATR."""
    _validate_positive(params.account_size, field_name="account_size")
    _validate_fraction(params.risk_percent, field_name="risk_percent")
    _validate_positive(params.entry_price, field_name="entry_price")
    _validate_positive(params.atr_multiplier, field_name="atr_multiplier")
    _validate_positive(params.atr_period, field_name="atr_period")
    _validate_positive(params.max_leverage, field_name="max_leverage")
    _validate_positive(params.lot_size, field_name="lot_size")
    _validate_positive(params.minimum_quantity, field_name="minimum_quantity", allow_zero=True)
    _validate_positive(params.commission_per_share, field_name="commission_per_share", allow_zero=True)
    _validate_positive(params.slippage_per_share, field_name="slippage_per_share", allow_zero=True)
    _validate_positive(params.fixed_fees, field_name="fixed_fees", allow_zero=True)
    _validate_positive(params.step_delay_seconds, field_name="step_delay_seconds", allow_zero=True)

    if params.side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    if params.sizing_method not in {"auto", "fixed_stop", "percent_stop", "atr"}:
        raise ValueError(
            "sizing_method must be 'auto', 'fixed_stop', 'percent_stop', or 'atr'"
        )
    if params.minimum_quantity and params.minimum_quantity % params.lot_size != 0:
        raise ValueError("minimum_quantity must be a multiple of lot_size")
    if params.max_position_percent is not None:
        _validate_fraction(params.max_position_percent, field_name="max_position_percent")
    if params.stop_percent is not None:
        _validate_fraction(params.stop_percent, field_name="stop_percent")
        if params.stop_percent >= 1:
            raise ValueError("stop_percent must be < 1")
    if params.available_capital is not None:
        _validate_positive(params.available_capital, field_name="available_capital", allow_zero=True)

    explain = bool(params.explain_steps)
    delay = float(params.step_delay_seconds)

    method = _resolve_sizing_method(params)
    stop_price, stop_distance, atr_value, stop_notes = _determine_stop_and_risk(params, method)

    _emit_step(
        explain,
        "Step 1: Stop derivation",
        (
            f"Method={method} | entry={params.entry_price:,.4f} | stop={stop_price:,.4f} | "
            f"stop_distance=|entry-stop|={stop_distance:,.4f}"
        ),
        delay_seconds=delay,
    )

    # Treat the friction inputs as per-side costs and convert to a conservative
    # round-trip per-share risk buffer.
    per_share_friction = 2.0 * (params.commission_per_share + params.slippage_per_share)
    risk_per_share = stop_distance + per_share_friction
    if risk_per_share <= 0:
        raise ValueError("risk_per_share must be > 0")

    _emit_step(
        explain,
        "Step 2: Per-share risk",
        (
            f"friction=2*(commission+slippage)=2*({params.commission_per_share:,.4f}+{params.slippage_per_share:,.4f})="
            f"{per_share_friction:,.4f}; risk_per_share=stop_distance+friction={stop_distance:,.4f}+{per_share_friction:,.4f}="
            f"{risk_per_share:,.4f}"
        ),
        delay_seconds=delay,
    )

    risk_amount = params.account_size * params.risk_percent
    risk_budget_after_fees = max(risk_amount - params.fixed_fees, 0.0)

    raw_quantity_by_risk = 0
    if risk_budget_after_fees > 0:
        raw_quantity_by_risk = math.floor(risk_budget_after_fees / risk_per_share)
    quantity_by_risk = _floor_to_lot(raw_quantity_by_risk, params.lot_size)

    _emit_step(
        explain,
        "Step 3: Risk budget to quantity",
        (
            f"risk_amount=account_size*risk_percent={params.account_size:,.2f}*{params.risk_percent:.4f}={risk_amount:,.2f}; "
            f"after_fees=max(risk_amount-fixed_fees,0)=max({risk_amount:,.2f}-{params.fixed_fees:,.2f},0)={risk_budget_after_fees:,.2f}; "
            f"raw_qty_by_risk=floor(after_fees/risk_per_share)={raw_quantity_by_risk}; lot-adjusted={quantity_by_risk}"
        ),
        delay_seconds=delay,
    )

    available_capital = (
        params.available_capital
        if params.available_capital is not None
        else params.account_size * params.max_leverage
    )

    max_position_value = available_capital
    if params.max_position_percent is not None:
        max_position_value = min(max_position_value, params.account_size * params.max_position_percent)

    raw_quantity_by_capital = math.floor(max_position_value / params.entry_price) if max_position_value > 0 else 0
    quantity_by_capital = _floor_to_lot(raw_quantity_by_capital, params.lot_size)

    _emit_step(
        explain,
        "Step 4: Capital cap to quantity",
        (
            f"available_capital={(params.available_capital if params.available_capital is not None else params.account_size * params.max_leverage):,.2f}; "
            f"max_position_value={max_position_value:,.2f}; raw_qty_by_capital=floor(max_position_value/entry)={raw_quantity_by_capital}; "
            f"lot-adjusted={quantity_by_capital}"
        ),
        delay_seconds=delay,
    )

    quantity = min(quantity_by_risk, quantity_by_capital)
    notes = list(stop_notes)

    if quantity < params.minimum_quantity:
        if quantity > 0:
            notes.append(
                f"Computed quantity {quantity} is below minimum_quantity "
                f"{params.minimum_quantity}; trade rejected"
            )
        quantity = 0

    position_value = quantity * params.entry_price
    actual_risk_amount = (
        quantity * risk_per_share + params.fixed_fees if quantity > 0 else 0.0
    )
    actual_risk_percent = actual_risk_amount / params.account_size if quantity > 0 else 0.0

    if quantity == 0:
        if raw_quantity_by_risk == 0:
            notes.append("Risk budget is too small for even one lot")
        if quantity_by_capital == 0:
            notes.append("Capital cap is too small for even one lot")

    binding_constraint = _binding_constraint(
        quantity,
        quantity_by_risk=quantity_by_risk,
        quantity_by_capital=quantity_by_capital,
        minimum_quantity=params.minimum_quantity,
    )

    _emit_step(
        explain,
        "Step 5: Final decision",
        (
            f"quantity=min(qty_by_risk, qty_by_capital)={quantity}; position_value=quantity*entry={position_value:,.2f}; "
            f"actual_risk=quantity*risk_per_share+fixed_fees={actual_risk_amount:,.2f}; binding={binding_constraint}"
        ),
        delay_seconds=delay,
    )

    return PositionSizingResult(
        ticker=params.ticker,
        risk_amount=_round_amount(risk_amount),
        risk_per_share=_round_price(risk_per_share),
        quantity=int(quantity),
        position_value=_round_amount(position_value),
        method=method,
        atr_value=_round_price(atr_value) if atr_value is not None else None,
        side=params.side,
        entry_price=_round_price(params.entry_price),
        stop_price=_round_price(stop_price),
        stop_distance=_round_price(stop_distance),
        quantity_by_risk=int(quantity_by_risk),
        quantity_by_capital=int(quantity_by_capital),
        raw_quantity=int(min(raw_quantity_by_risk, raw_quantity_by_capital)),
        max_position_value=_round_amount(max_position_value),
        actual_risk_amount=_round_amount(actual_risk_amount),
        actual_risk_percent=round(float(actual_risk_percent), 6),
        binding_constraint=binding_constraint,
        notes=tuple(notes),
    )


__all__ = [
    "PositionSizingInput",
    "PositionSizingResult",
    "calculate_position_size",
    "compute_true_range",
    "compute_atr",
]
