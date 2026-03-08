from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSizingInput:
    ticker: str
    account_size: float
    risk_percent: float
    entry_price: float
    stop_price: float


@dataclass(frozen=True)
class PositionSizingResult:
    ticker: str
    risk_amount: float
    risk_per_share: float
    quantity: int
    position_value: float



def calculate_position_size(params: PositionSizingInput) -> PositionSizingResult:
    """Simple fixed-risk position sizing.

    quantity = floor((account_size * risk_percent) / abs(entry - stop))
    """
    if params.account_size <= 0:
        raise ValueError("account_size must be > 0")
    if not (0 < params.risk_percent <= 1):
        raise ValueError("risk_percent must be in (0, 1]")
    if params.entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if params.stop_price <= 0:
        raise ValueError("stop_price must be > 0")

    risk_per_share = abs(params.entry_price - params.stop_price)
    if risk_per_share == 0:
        raise ValueError("entry_price and stop_price must be different")

    risk_amount = params.account_size * params.risk_percent
    quantity = int(risk_amount // risk_per_share)
    position_value = quantity * params.entry_price

    return PositionSizingResult(
        ticker=params.ticker,
        risk_amount=risk_amount,
        risk_per_share=risk_per_share,
        quantity=quantity,
        position_value=position_value,
    )
