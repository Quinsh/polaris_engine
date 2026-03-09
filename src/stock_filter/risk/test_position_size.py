import importlib.util
import sys

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "risk_engine.position_size",
    "position_size.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

P = mod.PositionSizingInput
f = mod.calculate_position_size


def run() -> None:
    r = f(P("AAPL", 100_000, 0.01, 50, stop_price=47))
    assert r.quantity == 333
    assert r.method == "fixed_stop"
    assert r.binding_constraint == "risk"

    r = f(P("AAPL", 10_000, 0.02, 100, stop_price=99, max_position_percent=0.1))
    assert r.quantity == 10
    assert r.binding_constraint == "capital"

    r = f(P("TSLA", 50_000, 0.02, 40, side="short", stop_price=42))
    assert r.quantity == 500
    assert r.stop_price == 42.0

    r = f(P("MSFT", 25_000, 0.01, 51, stop_price=48, lot_size=10))
    assert r.quantity == 80

    r = f(P("META", 100_000, 0.01, 100, stop_percent=0.02))
    assert r.method == "percent_stop"
    assert r.stop_price == 98.0
    assert r.quantity == 500

    df = pd.DataFrame(
        [{"HIGH": 101 + i * 0.1, "LOW": 99 + i * 0.1, "CLOSE": 100 + i * 0.1} for i in range(30)]
    )
    r = f(P("NVDA", 100_000, 0.01, 110, ohlcv=df, atr_period=14, atr_multiplier=2))
    assert r.method == "atr"
    assert r.atr_value is not None and r.atr_value > 0
    assert r.quantity > 0

    try:
        f(P("BAD", 10_000, 0.01, 50, stop_price=51))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    r = f(P("IBM", 10_000, 0.01, 100, stop_price=99, fixed_fees=15))
    assert r.quantity == 85

    print("All tests passed.")


if __name__ == "__main__":
    run()
