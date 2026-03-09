from __future__ import annotations

from stock_filter.risk import PositionSizingInput, calculate_position_size


def test_calculate_position_size_explain_steps_outputs(capsys) -> None:
    calculate_position_size(
        PositionSizingInput(
            ticker="005930",
            account_size=10_000_000,
            risk_percent=0.01,
            entry_price=70_000,
            stop_price=68_000,
            explain_steps=True,
            step_delay_seconds=0.0,
        )
    )
    out = capsys.readouterr().out
    assert "Step 1: Stop derivation" in out
    assert "Step 5: Final decision" in out
