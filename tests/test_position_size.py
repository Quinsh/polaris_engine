from __future__ import annotations

import pytest

from stock_filter.risk import PositionSizingInput, calculate_position_size


def test_calculate_position_size_basic_case() -> None:
    result = calculate_position_size(
        PositionSizingInput(
            ticker="005930",
            account_size=10_000_000,
            risk_percent=0.01,
            entry_price=70_000,
            stop_price=68_000,
        )
    )

    assert result.ticker == "005930"
    assert result.risk_amount == 100_000
    assert result.risk_per_share == 2_000
    assert result.quantity == 50
    assert result.position_value == 3_500_000


def test_calculate_position_size_rejects_same_entry_and_stop() -> None:
    with pytest.raises(ValueError, match="must be different"):
        calculate_position_size(
            PositionSizingInput(
                ticker="005930",
                account_size=10_000_000,
                risk_percent=0.01,
                entry_price=70_000,
                stop_price=70_000,
            )
        )
