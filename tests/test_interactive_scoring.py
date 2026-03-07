from __future__ import annotations

from stock_filter.app.interactive import average_signal_score


def test_average_signal_score_uses_all_selected_filters() -> None:
    signal_scores = {"vcp": 0.9, "double_bottom": 0.7, "rsi_mfi_smart_buy": 8.0}
    selected = ["vcp", "double_bottom", "rsi_mfi_smart_buy"]

    assert average_signal_score(signal_scores, selected) == (0.9 + 0.7 + 8.0) / 3


def test_average_signal_score_returns_zero_without_selection() -> None:
    assert average_signal_score({"vcp": 1.0}, []) == 0.0
