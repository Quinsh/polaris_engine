# Chart-pattern signals: VCP, double bottom, bullish pennant, etc.
# Each detector receives OHLCV (+ optional features) and returns Signal | None.
# See docs/SCREENING_STRUCTURE.md for where this fits in the pipeline.

#
# DataFrame structure (df):
#   Rows: one per bar (e.g. one per trading day), sorted by date ascending.
#         Row 0 = oldest date, last row = most recent (iloc[-1] = latest bar).
#   Columns (always present): ticker, date, open, high, low, close, volume.
#   Optional: value (거래대금), change (등락률), plus any feature columns
#             (e.g. returns_1d, volume_sma_20, price_above_sma_200) if requested.


from __future__ import annotations

import pandas as pd
import numpy as np

from stock_filter.analytics.types import Signal
from stock_filter.signals.registry import register_signal


def _last_row_info(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return "", ""
    row = df.iloc[-1]
    ticker = str(row.get("ticker", ""))
    asof = pd.Timestamp(row["date"]).strftime("%Y-%m-%d") if "date" in df.columns else ""
    return ticker, asof

### VCP Analysis Start #################################################

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(lo if x < lo else hi if x > hi else x)

def _vcp_finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")

def _vcp_finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")

def _vcp_score_smaller_better(x: float, good: float, bad: float) -> float:
    """Return 1 when x<=good, 0 when x>=bad, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if good >= bad:
        return 1.0 if x <= good else 0.0
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return float((bad - x) / (bad - good))

def _vcp_score_larger_better(x: float, bad: float, good: float) -> float:
    """Return 0 when x<=bad, 1 when x>=good, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if bad >= good:
        return 1.0 if x >= good else 0.0
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return float((x - bad) / (good - bad))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _zigzag_pivots(prices: np.ndarray, threshold: float) -> list[int]:
    """
    Percent zigzag pivots on 1D price array.
    threshold is a fraction (e.g. 0.05 for 5%).
    Returns pivot indices in chronological order.
    """
    n = int(len(prices))
    if n < 3 or not np.isfinite(threshold) or threshold <= 0:
        return [0, max(0, n - 1)]

    thr = float(threshold)
    trend = 0  # 0 unknown, 1 up, -1 down
    hi = lo = float(prices[0])
    hi_i = lo_i = 0
    pivots: list[int] = []

    for i in range(1, n):
        p = float(prices[i])
        if not np.isfinite(p):
            continue

        if p > hi:
            hi = p
            hi_i = i
        if p < lo:
            lo = p
            lo_i = i

        if trend == 0:
            # Establish initial trend only after a meaningful reversal from the most recent extreme.
            if hi_i > lo_i and hi > 0 and (hi - p) / hi >= thr:
                pivots.append(hi_i)  # first pivot is a high
                trend = -1
                lo = p
                lo_i = i
                hi = p
                hi_i = i
            elif lo_i > hi_i and lo > 0 and (p - lo) / lo >= thr:
                pivots.append(lo_i)  # first pivot is a low
                trend = 1
                hi = p
                hi_i = i
                lo = p
                lo_i = i

        elif trend == 1:
            # In uptrend: update high; confirm pivot high on sufficient drop.
            if p > hi:
                hi = p
                hi_i = i
            elif hi > 0 and (hi - p) / hi >= thr:
                pivots.append(hi_i)
                trend = -1
                lo = p
                lo_i = i
                hi = p
                hi_i = i

        else:  # trend == -1
            # In downtrend: update low; confirm pivot low on sufficient rally.
            if p < lo:
                lo = p
                lo_i = i
            elif lo > 0 and (p - lo) / lo >= thr:
                pivots.append(lo_i)
                trend = 1
                hi = p
                hi_i = i
                lo = p
                lo_i = i

    # Append last extreme so the most recent leg is represented.
    if trend == 1:
        pivots.append(hi_i)
    elif trend == -1:
        pivots.append(lo_i)
    else:
        pivots = [0, n - 1]

    # Clean: ensure strictly increasing and de-duplicate.
    cleaned: list[int] = []
    last = -1
    for idx in pivots:
        idx = int(idx)
        if idx <= last:
            continue
        cleaned.append(idx)
        last = idx

    if not cleaned:
        return [0, n - 1]
    if cleaned[0] != 0:
        cleaned.insert(0, 0)
    if cleaned[-1] != n - 1:
        cleaned.append(n - 1)

    return cleaned

def _label_pivots(prices: np.ndarray, pivots: list[int]) -> tuple[list[int], list[str]]:
    """Label pivots as 'H' or 'L' via neighbor comparison, then enforce alternation."""
    if len(pivots) < 2:
        return pivots, ["?"] * len(pivots)

    px = [float(prices[i]) for i in pivots]
    types: list[str] = []
    for j in range(len(pivots)):
        if j == 0:
            types.append("H" if px[0] >= px[1] else "L")
        elif j == len(pivots) - 1:
            types.append("H" if px[-1] >= px[-2] else "L")
        else:
            if px[j] >= px[j - 1] and px[j] >= px[j + 1]:
                types.append("H")
            elif px[j] <= px[j - 1] and px[j] <= px[j + 1]:
                types.append("L")
            else:
                types.append("H" if px[j] >= px[j - 1] else "L")

    # Merge consecutive same-type pivots (keep more extreme)
    out_idx: list[int] = [pivots[0]]
    out_typ: list[str] = [types[0]]
    for j in range(1, len(pivots)):
        idx = pivots[j]
        typ = types[j]
        if typ == out_typ[-1]:
            prev_idx = out_idx[-1]
            if typ == "H":
                if prices[idx] >= prices[prev_idx]:
                    out_idx[-1] = idx
            else:
                if prices[idx] <= prices[prev_idx]:
                    out_idx[-1] = idx
        else:
            out_idx.append(idx)
            out_typ.append(typ)

    return out_idx, out_typ


@register_signal("vcp", required_features=[])
def vcp(df: pd.DataFrame) -> Signal | None:
    """
    Volatility Contraction Pattern (VCP) scorer.

    Goal: return a 0–1 score for "how VCP-like" the *recent* structure is, using only OHLCV.
    This is a weighted scoring model over:
      - Stage-2 / uptrend context (MA alignment + proximity to highs)
      - 2–6 progressively smaller pullback contractions (zigzag swing model)
      - Volatility contraction (normalized ATR + realized range compression)
      - Volume contraction / dry-up on the right side
      - Pivot proximity (or breakout participation) at the last contraction pivot

    Searches patterns within the last MAX_LOOKBACK bars ("n days back").
    Returns None if the best pattern score is below a minimum quality threshold.
    """
    if df is None or df.empty:
        return None

    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date")
    d = d.reset_index(drop=True)

    # Need enough bars for a base + ATR + trend context
    if len(d) < 60:
        return None

    # "n days back" horizon
    MAX_LOOKBACK = 260  # ~1 year of dailies; adjust if you want longer/shorter scan
    tail = d.iloc[-min(len(d), MAX_LOOKBACK):].copy()

    for col in ["high", "low", "close", "volume"]:
        tail[col] = pd.to_numeric(tail[col], errors="coerce")
    if tail[["high", "low", "close", "volume"]].isna().any().any():
        return None

    high = tail["high"]
    low = tail["low"]
    close = tail["close"]
    volume = tail["volume"]

    # Indicators
    atr14 = _atr(high, low, close, n=14)
    natr14 = atr14 / close.replace(0, np.nan)

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()

    vma10 = volume.rolling(10, min_periods=10).mean()
    vma50 = volume.rolling(50, min_periods=50).mean()

    lookback_52 = int(min(252, len(tail)))
    hh_52 = high.rolling(lookback_52, min_periods=min(60, lookback_52)).max()

    # Latest values
    close_last = float(close.iloc[-1])
    vol_last = float(volume.iloc[-1])
    end_idx = len(tail) - 1

    # ---------------------------
    # 1) Trend / Stage-2 context
    # ---------------------------
    trend_components: list[float] = []
    trend_weights: list[float] = []

    def _add_component(val: float, w: float) -> None:
        if np.isfinite(val) and w > 0:
            trend_components.append(float(val))
            trend_weights.append(float(w))

    sma50_last = float(sma50.iloc[-1]) if np.isfinite(float(sma50.iloc[-1])) else float("nan")
    sma20_last = float(sma20.iloc[-1]) if np.isfinite(float(sma20.iloc[-1])) else float("nan")
    sma200_last = float(sma200.iloc[-1]) if np.isfinite(float(sma200.iloc[-1])) else float("nan")

    if np.isfinite(sma50_last):
        _add_component(1.0 if close_last > sma50_last else 0.0, 0.25)

        # Avoid "too extended" above short MA (late / blow-off)
        if np.isfinite(sma20_last) and sma20_last > 0:
            ext20 = (close_last / sma20_last) - 1.0
            _add_component(_vcp_score_smaller_better(ext20, good=0.02, bad=0.18), 0.08)

    if np.isfinite(sma200_last):
        _add_component(1.0 if close_last > sma200_last else 0.0, 0.10)
        if np.isfinite(sma50_last):
            _add_component(1.0 if sma50_last > sma200_last else 0.0, 0.15)

        # Slope proxies (20-bar % change)
        if len(sma50) >= 21 and np.isfinite(float(sma50.iloc[-21])) and float(sma50.iloc[-21]) > 0:
            slope50 = (float(sma50.iloc[-1]) / float(sma50.iloc[-21])) - 1.0
            _add_component(_vcp_score_larger_better(slope50, bad=-0.02, good=0.02), 0.10)
        if len(sma200) >= 21 and np.isfinite(float(sma200.iloc[-21])) and float(sma200.iloc[-21]) > 0:
            slope200 = (float(sma200.iloc[-1]) / float(sma200.iloc[-21])) - 1.0
            _add_component(_vcp_score_larger_better(slope200, bad=-0.01, good=0.01), 0.05)

    # Near highs (typical VCP posture)
    hh_last = float(hh_52.iloc[-1]) if np.isfinite(float(hh_52.iloc[-1])) else float("nan")
    if np.isfinite(hh_last) and hh_last > 0:
        near_high = close_last / hh_last
        _add_component(_vcp_score_larger_better(near_high, bad=0.75, good=0.92), 0.27)

    # Liquidity proxy: 20-day average dollar volume
    dollar_vol = (close * volume).rolling(20, min_periods=20).mean()
    dv_last = float(dollar_vol.iloc[-1]) if np.isfinite(float(dollar_vol.iloc[-1])) else float("nan")
    if np.isfinite(dv_last):
        _add_component(_vcp_score_larger_better(dv_last, bad=5e5, good=2e7), 0.10)

    trend_score = float(np.average(trend_components, weights=trend_weights)) if trend_weights else 0.0

    # -----------------------------------------
    # 2) Detect contractions via zigzag swings
    # -----------------------------------------
    ANALYSIS_WIN = int(min(len(tail), 200))  # where we search for the swing structure
    w_start = len(tail) - ANALYSIS_WIN
    w = tail.iloc[w_start:].reset_index(drop=True)

    w_close = w["close"].to_numpy(dtype=float)
    if not np.isfinite(w_close).all():
        return None

    # Dynamic zigzag threshold: based on recent normalized ATR (fallback to 5%)
    recent_natr = natr14.iloc[-min(60, len(natr14)) :].to_numpy(dtype=float)
    recent_natr = recent_natr[np.isfinite(recent_natr) & (recent_natr > 0)]
    natr_med = float(np.median(recent_natr)) if recent_natr.size else float("nan")
    zz_thr = 0.05 if not np.isfinite(natr_med) else float(np.clip(1.6 * natr_med, 0.03, 0.15))

    piv = _zigzag_pivots(w_close, threshold=zz_thr)
    piv, piv_type = _label_pivots(w_close, piv)

    # Build "contractions" as downswings (High -> Low) above a minimum meaningful depth
    downswings = []
    for j in range(len(piv) - 1):
        if piv_type[j] != "H" or piv_type[j + 1] != "L":
            continue

        hi_w = int(piv[j])
        lo_w = int(piv[j + 1])
        if lo_w <= hi_w:
            continue

        hi_t = w_start + hi_w
        lo_t = w_start + lo_w

        hi_price = float(high.iloc[hi_t])
        lo_price = float(low.iloc[lo_t])
        if not np.isfinite(hi_price) or not np.isfinite(lo_price) or hi_price <= 0:
            continue

        depth = (hi_price - lo_price) / hi_price
        if not np.isfinite(depth) or depth <= 0:
            continue

        # Ignore micro-swings: require depth to exceed ~1.2x typical normalized ATR around the swing high (clipped)
        local_natr = _vcp_finite_median(natr14.iloc[max(0, hi_t - 20) : hi_t + 1].to_numpy(dtype=float))
        min_depth = 0.02 if not np.isfinite(local_natr) else float(np.clip(1.2 * local_natr, 0.02, 0.08))
        if depth < min_depth:
            continue

        seg = slice(hi_t, lo_t + 1)
        avg_vol = float(volume.iloc[seg].mean())
        avg_natr_seg = _vcp_finite_mean(natr14.iloc[seg].to_numpy(dtype=float))

        downswings.append(
            {
                "hi_idx": hi_t,
                "lo_idx": lo_t,
                "hi_price": hi_price,
                "lo_price": lo_price,
                "depth": float(depth),
                "bars": int(lo_t - hi_t),
                "avg_vol": avg_vol,
                "avg_natr": float(avg_natr_seg),
            }
        )

    if len(downswings) < 2:
        return None

    # -------------------------------------------------------
    # 3) Evaluate candidate sequences (2–6 contractions)
    #    Choose the best score among the last-k downswings.
    # -------------------------------------------------------
    best: dict | None = None

    for k in range(2, min(6, len(downswings)) + 1):
        seq = downswings[-k:]
        depths = np.array([s["depth"] for s in seq], dtype=float)
        hi_prices = np.array([s["hi_price"] for s in seq], dtype=float)
        lo_prices = np.array([s["lo_price"] for s in seq], dtype=float)

        base_start = int(seq[0]["hi_idx"])
        base_end = end_idx
        if base_end <= base_start:
            continue
        base_len = int(base_end - base_start + 1)

        # Depth ratios between contractions: want <1 on average (progressively smaller)
        ratios = depths[1:] / np.maximum(depths[:-1], 1e-12)
        mean_ratio = float(np.mean(ratios)) if ratios.size else float("nan")
        monotonic_score = _vcp_score_smaller_better(mean_ratio, good=0.75, bad=1.10)

        contraction_ratio = float(depths[-1] / max(depths[0], 1e-12))
        ratio_score = _vcp_score_smaller_better(contraction_ratio, good=0.35, bad=0.85)

        tight_score = _vcp_score_smaller_better(float(depths[-1]), good=0.04, bad=0.12)
        count_score = _vcp_score_larger_better(float(k), bad=2.0, good=4.0)

        # Converging character: highs should not be rising; lows ideally rising
        highs_change = (float(hi_prices[-1]) / float(hi_prices[0])) - 1.0 if hi_prices[0] > 0 else float("nan")
        lows_change = (float(lo_prices[-1]) / float(lo_prices[0])) - 1.0 if lo_prices[0] > 0 else float("nan")
        highs_trend_score = _vcp_score_smaller_better(1.0 + highs_change, good=1.00, bad=1.05)
        lows_trend_score = _vcp_score_larger_better(lows_change, bad=-0.06, good=0.02)

        # Base length: avoid too-short noise; very long bases get a mild penalty (late/loose)
        length_score = _vcp_score_larger_better(float(base_len), bad=18.0, good=70.0)
        if base_len > 170:
            length_score *= 0.6

        # Volatility contraction (normalized ATR) early vs late
        early_slice = slice(base_start, base_start + max(10, base_len // 2))
        late_slice = slice(max(base_start, base_end - max(10, base_len // 3)), base_end + 1)
        early_natr = _vcp_finite_mean(natr14.iloc[early_slice].to_numpy(dtype=float))
        late_natr = _vcp_finite_mean(natr14.iloc[late_slice].to_numpy(dtype=float))
        natr_ratio = (late_natr / max(early_natr, 1e-12)) if np.isfinite(early_natr) else float("nan")
        vol_contraction_score = _vcp_score_smaller_better(natr_ratio, good=0.65, bad=1.05)

        # Realized range contraction early vs late
        early_rng = float((high.iloc[early_slice].max() - low.iloc[early_slice].min()) / max(close.iloc[early_slice].median(), 1e-12))
        late_rng = float((high.iloc[late_slice].max() - low.iloc[late_slice].min()) / max(close.iloc[late_slice].median(), 1e-12))
        rng_ratio = (late_rng / max(early_rng, 1e-12)) if np.isfinite(early_rng) and early_rng > 0 else float("nan")
        range_contraction_score = _vcp_score_smaller_better(rng_ratio, good=0.55, bad=1.05)

        # Volume contraction early vs late (exclude last 1–2 bars from late avg to avoid breakout bar contamination)
        early_vol = float(volume.iloc[early_slice].mean())
        late_vol_slice = slice(max(base_start, base_end - max(12, base_len // 3)), max(base_start + 1, base_end - 1))
        late_vol = float(volume.iloc[late_vol_slice].mean()) if late_vol_slice.stop > late_vol_slice.start else float(volume.iloc[late_slice].mean())
        vol_ratio = (late_vol / max(early_vol, 1e-12)) if early_vol > 0 else float("nan")
        volume_contraction_score = _vcp_score_smaller_better(vol_ratio, good=0.60, bad=1.10)

        vma50_last = float(vma50.iloc[-1]) if np.isfinite(float(vma50.iloc[-1])) else float("nan")
        dry_slice = slice(max(base_start, base_end - 7), max(base_start + 1, base_end - 2))
        dry_avg = float(volume.iloc[dry_slice].mean()) if dry_slice.stop > dry_slice.start else float("nan")
        dry_ratio = (dry_avg / max(vma50_last, 1e-12)) if np.isfinite(vma50_last) and vma50_last > 0 else float("nan")
        dryup_score = _vcp_score_smaller_better(dry_ratio, good=0.55, bad=1.30)

        # "Tight closes" (absorption): closes near top of the day on the right side
        tight_close_slice = slice(max(base_start, base_end - 10), base_end + 1)
        rng = (high.iloc[tight_close_slice] - low.iloc[tight_close_slice]).replace(0, np.nan)
        close_pos = ((close.iloc[tight_close_slice] - low.iloc[tight_close_slice]) / rng).clip(lower=0.0, upper=1.0)
        tight_close = float(close_pos.mean()) if close_pos.notna().any() else float("nan")
        tight_close_score = _vcp_score_larger_better(tight_close, bad=0.35, good=0.65)

        # Pivot / trigger readiness: pivot = high of the last (right-most) contraction
        pivot_price = float(seq[-1]["hi_price"])
        pivot_dist = (pivot_price - close_last) / pivot_price if pivot_price > 0 else float("nan")
        breakout = bool(np.isfinite(pivot_dist) and pivot_dist < 0)

        if breakout:
            # Fresh breakouts: small extension above pivot + decent participation
            ext = (close_last / pivot_price) - 1.0
            ext_score = _vcp_score_smaller_better(ext, good=0.02, bad=0.14)
            vol_part = (vol_last / vma50_last) if np.isfinite(vma50_last) and vma50_last > 0 else float("nan")
            vol_part_score = _vcp_score_larger_better(vol_part, bad=0.9, good=1.6)
            pivot_score = ext_score * vol_part_score
        else:
            # Pre-breakout setups: very close under pivot is best
            pivot_score = _vcp_score_smaller_better(pivot_dist, good=0.02, bad=0.10)

        # Prior run-up into base: separate VCP (continuation) from bottoming patterns
        pre_idx = max(0, base_start - 60)
        if base_start - pre_idx >= 10 and float(close.iloc[pre_idx]) > 0:
            runup = (float(close.iloc[base_start]) / float(close.iloc[pre_idx])) - 1.0
            runup_score = _vcp_score_larger_better(runup, bad=0.00, good=0.25)
        else:
            runup = float("nan")
            runup_score = 0.5  # neutral if unmeasurable

        # Combine into final score
        contraction_structure = float(
            np.average(
                [
                    monotonic_score,
                    ratio_score,
                    tight_score,
                    count_score,
                    highs_trend_score,
                    lows_trend_score,
                    length_score,
                ],
                weights=[0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.15],
            )
        )

        behavior_score = float(
            np.average(
                [
                    vol_contraction_score,
                    range_contraction_score,
                    volume_contraction_score,
                    dryup_score,
                    tight_close_score,
                ],
                weights=[0.25, 0.15, 0.25, 0.15, 0.20],
            )
        )

        total_score = float(
            np.average(
                [trend_score, runup_score, contraction_structure, behavior_score, pivot_score],
                weights=[0.25, 0.10, 0.35, 0.20, 0.10],
            )
        )

        # Hard downweights for anti-VCP context
        if np.isfinite(sma50_last) and close_last <= sma50_last:
            total_score *= 0.55
        if np.isfinite(natr_ratio) and natr_ratio > 1.15:
            total_score *= 0.75

        total_score = _clamp(total_score, 0.0, 1.0)

        if best is None or total_score > float(best["score"]):
            best = {
                "score": float(total_score),
                "k": int(k),
                "base_start": int(base_start),
                "base_end": int(base_end),
                "base_len": int(base_len),
                "zz_thr": float(zz_thr),
                "depths": [float(x) for x in depths.tolist()],
                "mean_depth_ratio": float(mean_ratio) if np.isfinite(mean_ratio) else None,
                "contraction_ratio": float(contraction_ratio),
                "pivot": float(pivot_price),
                "pivot_dist": float(pivot_dist) if np.isfinite(pivot_dist) else None,
                "breakout": bool(breakout),
                "trend_score": float(trend_score),
                "runup": float(runup) if np.isfinite(runup) else None,
                "runup_score": float(runup_score),
                "structure_score": float(contraction_structure),
                "behavior_score": float(behavior_score),
                "pivot_score": float(pivot_score),
                "volume_contraction": float(vol_ratio) if np.isfinite(vol_ratio) else None,
                "dryup_ratio": float(dry_ratio) if np.isfinite(dry_ratio) else None,
                "natr_ratio": float(natr_ratio) if np.isfinite(natr_ratio) else None,
                "rng_ratio": float(rng_ratio) if np.isfinite(rng_ratio) else None,
                "hi_idxs": [int(s["hi_idx"]) for s in seq],
                "lo_idxs": [int(s["lo_idx"]) for s in seq],
            }

    if best is None:
        return None

    # Emit only if there's a meaningful VCP-like match
    MIN_SCORE = 0.38
    if float(best["score"]) < MIN_SCORE or float(best["structure_score"]) < 0.35:
        return None

    # Optional: include human-readable dates in details if available
    if "date" in tail.columns:
        try:
            best["base_start_date"] = str(tail["date"].iloc[int(best["base_start"])])
            best["base_end_date"] = str(tail["date"].iloc[int(best["base_end"])])
        except Exception:
            pass

    ticker, asof = _last_row_info(df)
    return Signal(
        name="vcp",
        ticker=ticker,
        asof=asof,
        score=round(float(best["score"]), 4),
        details=best,
    )

### VCP Analysis End #################################################


### Double Bottom Start #################################################


def _db_score_smaller_better(x: float, good: float, bad: float) -> float:
    """1 when x <= good, 0 when x >= bad, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if good >= bad:
        return 1.0 if x <= good else 0.0
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return float((bad - x) / (bad - good))

def _db_score_larger_better(x: float, bad: float, good: float) -> float:
    """0 when x <= bad, 1 when x >= good, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if bad >= good:
        return 1.0 if x >= good else 0.0
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return float((x - bad) / (good - bad))


def _db_finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")



@register_signal("double_bottom", required_features=[])
def double_bottom(df: pd.DataFrame) -> Signal | None:
    """
    Double bottom (W) detector for "forming" patterns:
      - Structure: Low1 -> Peak (neckline) -> Low2 (similar depth)
      - Current price is on the right side (after Low2), rising or building toward neckline
      - Breakout not required; breakout (if present) increases score

    Returns Signal with score in [0, 1] or None.
    """
    if df is None or df.empty or len(df) < 30:
        return None

    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date")
    d = d.reset_index(drop=True)

    # More robust minimum for meaningful W detection (still respects your df>=30 guard above)
    if len(d) < 60:
        return None

    # Scan horizon: "today until n days back"
    MAX_LOOKBACK = 320
    tail = d.iloc[-min(len(d), MAX_LOOKBACK):].copy().reset_index(drop=True)

    for col in ["high", "low", "close", "volume"]:
        tail[col] = pd.to_numeric(tail[col], errors="coerce")
    if tail[["high", "low", "close", "volume"]].isna().any().any():
        return None

    high = tail["high"]
    low = tail["low"]
    close = tail["close"]
    volume = tail["volume"]

    end_idx = len(tail) - 1
    close_last = float(close.iloc[-1])
    high_last = float(high.iloc[-1])
    vol_last = float(volume.iloc[-1])

    # Indicators for thresholds / scoring
    atr14 = _atr(high, low, close, n=14)
    natr14 = atr14 / close.replace(0, np.nan)

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    vma10 = volume.rolling(10, min_periods=10).mean()
    vma50 = volume.rolling(50, min_periods=50).mean()

    # Dynamic zigzag threshold (ATR-based; fallback 5%)
    recent_natr = natr14.iloc[-min(80, len(natr14)) :].to_numpy(dtype=float)
    natr_med = _db_finite_median(recent_natr[recent_natr > 0])
    zz_thr = 0.05 if not np.isfinite(natr_med) else float(np.clip(1.6 * natr_med, 0.03, 0.14))

    # Run pivots on last ANALYSIS_WIN bars to focus on relevant structure
    ANALYSIS_WIN = int(min(len(tail), 220))
    w_start = len(tail) - ANALYSIS_WIN
    w = tail.iloc[w_start:].reset_index(drop=True)

    w_close = w["close"].to_numpy(dtype=float)
    piv = _zigzag_pivots(w_close, threshold=zz_thr)
    piv, typ = _label_pivots(w_close, piv)

    # Convert pivot indices to tail indices
    piv_t = [w_start + int(i) for i in piv]
    typ_t = typ[:]  # same length

    # Candidate selection settings
    MIN_SEP = 5                 # bars between L->H and H->L
    MAX_BARS_SINCE_LOW2 = 120   # ensures "currently forming"
    LOW_SIM_TOL = 0.07          # lows must be within ~7% (tunable)
    MIN_DEPTH = 0.06            # neckline must be at least ~6% above lows (tunable)

    best: dict | None = None

    # Find L-H-L sequences (W skeleton)
    for j in range(len(piv_t) - 2):
        if typ_t[j] != "L" or typ_t[j + 1] != "H" or typ_t[j + 2] != "L":
            continue

        i1 = int(piv_t[j])       # Low1 index in tail
        ih = int(piv_t[j + 1])   # Peak/neckline index in tail
        i2 = int(piv_t[j + 2])   # Low2 index in tail

        if not (0 <= i1 < ih < i2 <= end_idx):
            continue
        if (ih - i1) < MIN_SEP or (i2 - ih) < MIN_SEP:
            continue
        if (end_idx - i2) > MAX_BARS_SINCE_LOW2:
            continue

        low1 = float(low.iloc[i1])
        low2 = float(low.iloc[i2])
        neck = float(high.iloc[ih])
        if not (np.isfinite(low1) and np.isfinite(low2) and np.isfinite(neck)):
            continue
        if low1 <= 0 or low2 <= 0 or neck <= 0:
            continue

        # Similar lows
        low_min = min(low1, low2)
        low_diff = abs(low1 - low2) / low_min
        if low_diff > LOW_SIM_TOL:
            continue

        # Depth (neckline sufficiently above lows)
        depth = (neck - low_min) / neck
        if depth < MIN_DEPTH:
            continue

        # Right-side must be forming: price should be after Low2 and not collapsing below it
        post_min = float(low.iloc[i2 : end_idx + 1].min())
        undercut = (low2 - post_min) / low2  # positive if a later bar went below low2
        # allow small retest / tiny undercut; big undercut is "not forming"
        if post_min < low2 * (1.0 - 0.04):
            continue

        # Must be at least slightly off the second low (still "forming" if early)
        off_low2 = (close_last / low2) - 1.0
        if off_low2 < 0.006 and (end_idx - i2) > 6:
            # If low2 isn't extremely recent and we haven't lifted at all -> too early/weak
            continue

        # Stage / progress toward neckline (breakout not required)
        # Treat breakout if close > neckline OR intraday pierce meaningfully above neckline.
        breakout = (close_last > neck) or (high_last > neck * 1.01)
        progress = close_last / neck  # <1 => still below neckline; >1 => breakout/extension
        pivot_dist = (neck - close_last) / neck  # positive below neckline; negative above

        # Volume signals (optional scoring, not hard filter)
        # Prefer: vol around Low2 <= vol around Low1 (less selling on second low)
        wL = 4
        vol1 = float(volume.iloc[max(0, i1 - wL) : min(end_idx + 1, i1 + wL + 1)].mean())
        vol2 = float(volume.iloc[max(0, i2 - wL) : min(end_idx + 1, i2 + wL + 1)].mean())
        vol_ratio = (vol2 / vol1) if (vol1 > 0 and np.isfinite(vol1) and np.isfinite(vol2)) else float("nan")

        # Prior downtrend context (double bottom is typically reversal after decline)
        pre_start = max(0, i1 - 80)
        if i1 - pre_start >= 10 and float(close.iloc[pre_start]) > 0:
            drawdown = (float(close.iloc[pre_start]) - float(close.iloc[i1])) / float(close.iloc[pre_start])
        else:
            drawdown = float("nan")

        # Symmetry (time balance)
        left = ih - i1
        right = i2 - ih
        sym_ratio = (max(left, right) / max(1, min(left, right))) if min(left, right) > 0 else float("inf")

        # Right-leg momentum / upturn
        # Slope proxy: last 10 bars regression slope of close, normalized
        last_n = min(12, len(tail))
        y = close.iloc[-last_n:].to_numpy(dtype=float)
        x = np.arange(last_n, dtype=float)
        if last_n >= 8 and np.isfinite(y).all() and np.isfinite(y.mean()) and y.mean() > 0:
            # simple OLS slope
            x0 = x - x.mean()
            y0 = y - y.mean()
            denom = float(np.dot(x0, x0))
            slope = float(np.dot(x0, y0) / denom) if denom > 0 else 0.0
            slope_norm = slope / float(y.mean())
        else:
            slope_norm = float("nan")

        # ----------------
        # Component scores
        # ----------------
        similarity_score = _db_score_smaller_better(low_diff, good=0.012, bad=LOW_SIM_TOL)  # very similar => high
        depth_score = _db_score_larger_better(depth, bad=MIN_DEPTH, good=0.22)
        symmetry_score = _db_score_smaller_better(sym_ratio, good=1.6, bad=3.8)

        base_len = i2 - i1
        base_len_score = _db_score_larger_better(float(base_len), bad=18.0, good=70.0)
        if base_len > 180:
            base_len_score *= 0.65

        integrity_score = _db_score_smaller_better(max(0.0, undercut), good=0.0, bad=0.04)

        right_leg_score = _db_score_larger_better(off_low2, bad=0.01, good=0.18)
        upturn_score = _db_score_larger_better(slope_norm, bad=-0.002, good=0.004)

        # Neckline progress:
        # - below neckline: score rises as we approach it
        # - above neckline: penalize if too extended (late)
        if not np.isfinite(progress):
            neckline_score = 0.0
        elif progress < 1.0:
            neckline_score = _db_score_larger_better(progress, bad=0.74, good=0.985)
        else:
            ext = progress - 1.0
            neckline_score = _db_score_smaller_better(ext, good=0.02, bad=0.18)

        # Volume score:
        # - if breakout: want participation (vol_last / vma50)
        # - else: mild preference for quieting volume (vma10 < vma50) and/or vol2 <= vol1
        v50 = float(vma50.iloc[-1]) if np.isfinite(float(vma50.iloc[-1])) else float("nan")
        v10 = float(vma10.iloc[-1]) if np.isfinite(float(vma10.iloc[-1])) else float("nan")

        if breakout and np.isfinite(v50) and v50 > 0:
            part = vol_last / v50
            volume_score = _db_score_larger_better(part, bad=0.9, good=1.9)
        else:
            vol2_vs_vol1_score = _db_score_smaller_better(vol_ratio, good=0.72, bad=1.30) if np.isfinite(vol_ratio) else 0.5
            quiet_score = _db_score_smaller_better(v10 / v50, good=0.70, bad=1.10) if (np.isfinite(v10) and np.isfinite(v50) and v50 > 0) else 0.5
            volume_score = 0.55 * vol2_vs_vol1_score + 0.45 * quiet_score

        # Context (downtrend + reclaiming MAs)
        context_components = []
        context_weights = []

        if np.isfinite(drawdown):
            context_components.append(_db_score_larger_better(drawdown, bad=0.00, good=0.22))
            context_weights.append(0.55)
        else:
            context_components.append(0.5)
            context_weights.append(0.55)

        sma20_last = float(sma20.iloc[-1]) if np.isfinite(float(sma20.iloc[-1])) else float("nan")
        sma50_last = float(sma50.iloc[-1]) if np.isfinite(float(sma50.iloc[-1])) else float("nan")
        if np.isfinite(sma20_last) and sma20_last > 0:
            context_components.append(1.0 if close_last > sma20_last else 0.0)
            context_weights.append(0.25)
        if np.isfinite(sma50_last) and sma50_last > 0:
            context_components.append(_db_score_larger_better(close_last / sma50_last, bad=0.97, good=1.03))
            context_weights.append(0.20)

        context_score = float(np.average(context_components, weights=context_weights)) if context_weights else 0.5

        # Aggregate score
        # Structure dominates; then right-leg “forming” stage; then context/volume.
        structure_score = float(
            np.average(
                [similarity_score, depth_score, symmetry_score, base_len_score, integrity_score],
                weights=[0.28, 0.26, 0.14, 0.12, 0.20],
            )
        )

        forming_score = float(
            np.average(
                [right_leg_score, neckline_score, upturn_score],
                weights=[0.45, 0.35, 0.20],
            )
        )

        total_score = float(
            np.average(
                [structure_score, forming_score, context_score, volume_score],
                weights=[0.45, 0.35, 0.12, 0.08],
            )
        )

        # Small penalty if the pattern is extremely old relative to our “forming now” goal
        age2 = end_idx - i2
        if age2 > 90:
            total_score *= 0.85

        total_score = _clamp(total_score, 0.0, 1.0)

        cand = {
            "score": float(total_score),
            "zz_thr": float(zz_thr),
            "i_low1": int(i1),
            "i_neck": int(ih),
            "i_low2": int(i2),
            "low1": float(low1),
            "low2": float(low2),
            "neckline": float(neck),
            "low_diff": float(low_diff),
            "depth": float(depth),
            "sym_ratio": float(sym_ratio),
            "base_len": int(base_len),
            "bars_since_low2": int(age2),
            "post_low2_min": float(post_min),
            "undercut_after_low2": float(undercut),
            "off_low2": float(off_low2),
            "progress_to_neck": float(progress) if np.isfinite(progress) else None,
            "pivot_dist": float(pivot_dist) if np.isfinite(pivot_dist) else None,
            "breakout": bool(breakout),
            "drawdown_into_low1": float(drawdown) if np.isfinite(drawdown) else None,
            "vol_low2_vs_low1": float(vol_ratio) if np.isfinite(vol_ratio) else None,
            "components": {
                "structure": float(structure_score),
                "forming": float(forming_score),
                "context": float(context_score),
                "volume": float(volume_score),
                "similarity": float(similarity_score),
                "depth": float(depth_score),
                "symmetry": float(symmetry_score),
                "base_len": float(base_len_score),
                "integrity": float(integrity_score),
                "right_leg": float(right_leg_score),
                "neckline": float(neckline_score),
                "upturn": float(upturn_score),
            },
        }

        if "date" in tail.columns:
            try:
                cand["date_low1"] = str(tail["date"].iloc[i1])
                cand["date_neck"] = str(tail["date"].iloc[ih])
                cand["date_low2"] = str(tail["date"].iloc[i2])
            except Exception:
                pass

        if best is None or float(cand["score"]) > float(best["score"]):
            best = cand

    if best is None:
        return None

    # “Seemingly forming” gate:
    # Keep this moderate so you can rank candidates without requiring a breakout.
    MIN_SCORE = 0.38
    if float(best["score"]) < MIN_SCORE or float(best["components"]["structure"]) < 0.35:
        return None

    ticker, asof = _last_row_info(df)
    return Signal(
        name="double_bottom",
        ticker=ticker,
        asof=asof,
        score=round(float(best["score"]), 4),
        details=best,
    )


### DOUBLE BOTTOM END #################################################


### SUPPORT TEST #################################################
import numpy as np
import pandas as pd


def _st_score_smaller_better(x: float, good: float, bad: float) -> float:
    """1 when x <= good, 0 when x >= bad, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if good >= bad:
        return 1.0 if x <= good else 0.0
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return float((bad - x) / (bad - good))

def _st_score_larger_better(x: float, bad: float, good: float) -> float:
    """0 when x <= bad, 1 when x >= good, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if bad >= good:
        return 1.0 if x >= good else 0.0
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return float((x - bad) / (good - bad))

def _st_score_in_band(x: float, good_lo: float, good_hi: float, bad_lo: float, bad_hi: float) -> float:
    """
    Bell-ish score: 1 inside [good_lo, good_hi], 0 outside [bad_lo, bad_hi],
    linear ramps in-between.
    """
    if not np.isfinite(x):
        return 0.0
    if x < bad_lo or x > bad_hi:
        return 0.0
    if good_lo <= x <= good_hi:
        return 1.0
    if x < good_lo:
        # ramp from bad_lo -> good_lo
        return float((x - bad_lo) / max(1e-12, (good_lo - bad_lo)))
    # x > good_hi
    return float((bad_hi - x) / max(1e-12, (bad_hi - good_hi)))


def _st_finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")



@register_signal("support_test", required_features=[])
def support_test(df: pd.DataFrame) -> Signal | None:
    """
    Support trendline test (reversal-up intent):

    - Build a support line from TWO significant swing LOW pivots.
    - After the second pivot, NO candle low may go below that support line
      (otherwise that pivot pair is invalid).
    - Current close must be very close to the support line (testing it).
    - Pivots must be reasonably time-balanced: (pivot2 - pivot1) ~= (now - pivot2).
      Overly imbalanced spacing is rejected; closer-to-equal spacing scores higher.

    Input: OHLCV. Output: Signal(score in [0,1]) or None.
    """
    if df is None or df.empty or len(df) < 120:
        return None

    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date")
    d = d.reset_index(drop=True)

    MAX_LOOKBACK = 420
    tail = d.iloc[-min(len(d), MAX_LOOKBACK):].copy().reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in tail.columns:
            tail[col] = pd.to_numeric(tail[col], errors="coerce")
    if tail[["high", "low", "close", "volume"]].isna().any().any():
        return None

    high = tail["high"]
    low = tail["low"]
    close = tail["close"]
    volume = tail["volume"]

    end_idx = len(tail) - 1
    close_last = float(close.iloc[-1])
    high_last = float(high.iloc[-1])
    low_last = float(low.iloc[-1])
    open_last = float(tail["open"].iloc[-1]) if "open" in tail.columns and np.isfinite(float(tail["open"].iloc[-1])) else close_last

    # Volatility for dynamic pivot threshold and proximity tolerances
    atr14 = _atr(high, low, close, n=14)
    natr14 = atr14 / close.replace(0, np.nan)
    recent_natr = natr14.iloc[-min(120, len(natr14)) :].to_numpy(dtype=float)
    natr_med = _st_finite_median(recent_natr[(recent_natr > 0) & np.isfinite(recent_natr)])

    # Significant pivots: ATR-scaled zigzag threshold
    zz_thr = 0.05 if not np.isfinite(natr_med) else float(np.clip(1.6 * natr_med, 0.03, 0.14))

    # Proximity tolerances (fractions of price)
    # These are for "close to support" scoring, NOT for allowing line breaks.
    tol = 0.012 if not np.isfinite(natr_med) else float(np.clip(1.2 * natr_med, 0.006, 0.05))
    good_dist = float(np.clip(0.18 * tol, 0.0015, 0.010))
    bad_dist = float(np.clip(1.8 * tol, 0.012, 0.060))
    touch_tol = float(np.clip(0.55 * tol, 0.0025, 0.030))

    # Pivots on focused analysis window
    ANALYSIS_WIN = int(min(len(tail), 280))
    w_start = len(tail) - ANALYSIS_WIN
    w = tail.iloc[w_start:].reset_index(drop=True)

    w_close = w["close"].to_numpy(dtype=float)
    piv = _zigzag_pivots(w_close, threshold=zz_thr)
    piv, typ = _label_pivots(w_close, piv)
    piv_t = [w_start + int(i) for i in piv]
    typ_t = typ[:]

    low_pivots = [piv_t[i] for i in range(len(piv_t)) if typ_t[i] == "L"]
    if len(low_pivots) < 2:
        return None

    # Limit candidate search
    low_pivots = low_pivots[-26:]

    # Constraints
    MIN_SPAN_BARS = 20     # ~ 1 month
    MIN_AFTER_P2 = 6       # need some bars after pivot2 to validate "no break"
    MAX_IMBALANCE = 6.0    # hard reject if one segment is >6x the other
    MAX_FUTURE = 220       # ensure current test isn't absurdly far from pivots

    idx = np.arange(len(tail), dtype=float)
    best: dict | None = None

    for a in range(len(low_pivots) - 1):
        i1 = int(low_pivots[a])
        for b in range(a + 1, len(low_pivots)):
            i2 = int(low_pivots[b])

            d12 = i2 - i1
            d2t = end_idx - i2

            if d12 < MIN_SPAN_BARS:
                continue
            if d2t < MIN_AFTER_P2:
                continue
            if d2t > MAX_FUTURE:
                continue

            # Spacing balance (your "three pivots equidistant" preference)
            small = float(min(d12, d2t))
            big = float(max(d12, d2t))
            if small <= 0:
                continue
            spacing_ratio = big / small
            if spacing_ratio > MAX_IMBALANCE:
                continue

            p1 = float(low.iloc[i1])
            p2 = float(low.iloc[i2])
            if not (np.isfinite(p1) and np.isfinite(p2)) or p1 <= 0 or p2 <= 0:
                continue

            # Trendline through the two lows: line[t] = m*t + c
            m = (p2 - p1) / float(d12)
            c = p1 - m * float(i1)
            line = m * idx + c
            line_now = float(line[end_idx])
            if not np.isfinite(line_now) or line_now <= 0:
                continue

            # (1) HARD FILTER: after pivot2, no candle low may be below the line
            # Use an ultra-small epsilon only to avoid floating-point edge weirdness.
            eps = 1e-12
            seg = np.arange(i2 + 1, end_idx + 1)
            if seg.size > 0:
                lows_seg = low.iloc[seg].to_numpy(dtype=float)
                line_seg = line[seg]
                if np.any(lows_seg + eps < line_seg):
                    continue

            # Current "test": close must be close to line, from above (it must be above if no lows broke)
            dist_above = (close_last - line_now) / line_now  # >= 0 given the hard filter
            if not np.isfinite(dist_above) or dist_above < 0:
                continue
            if dist_above > bad_dist:
                continue

            # Extra: ensure the last bar is not wildly above line intraday (still "testing")
            if (high_last - line_now) / line_now > max(0.08, 5.0 * bad_dist):
                continue

            # Count touches AFTER pivot2 (excluding pivot2 itself and last bar)
            # touch = low within touch_tol of the line (still not below)
            touch_seg = np.arange(i2 + 1, max(i2 + 2, end_idx))  # up to end_idx-1
            if touch_seg.size > 0:
                low_touch = low.iloc[touch_seg].to_numpy(dtype=float)
                line_touch = line[touch_seg]
                touch_mask = (np.abs(low_touch - line_touch) / np.maximum(line_touch, 1e-12) <= touch_tol)
                touches = int(np.sum(touch_mask))
                min_low_dist = float(np.min((low_touch - line_touch) / np.maximum(line_touch, 1e-12)))
            else:
                touches = 0
                min_low_dist = float("nan")

            # Pivot significance: each pivot should have a meaningful bounce after it
            max_high_12 = float(high.iloc[i1 : i2 + 1].max())
            max_high_2t = float(high.iloc[i2 : end_idx + 1].max())
            bounce1 = (max_high_12 / p1) - 1.0 if p1 > 0 else float("nan")
            bounce2 = (max_high_2t / p2) - 1.0 if p2 > 0 else float("nan")
            pivot_sig = 0.5 * _st_score_larger_better(bounce1, bad=0.02, good=0.16) + \
                        0.5 * _st_score_larger_better(bounce2, bad=0.02, good=0.16)

            # Duration/age scoring (not short-term)
            span_score = _st_score_larger_better(float(d12), bad=float(MIN_SPAN_BARS), good=150.0)
            age_score = _st_score_larger_better(float(end_idx - i1), bad=45.0, good=260.0)
            duration_score = 0.55 * span_score + 0.45 * age_score

            # Spacing score (your requirement #2)
            spacing_score = _st_score_smaller_better(spacing_ratio, good=1.25, bad=3.5)

            # Proximity score (dominant)
            proximity_score = _st_score_smaller_better(dist_above, good=good_dist, bad=bad_dist)

            # "Actually testing" score: the closer the recent lows got to the line, the better
            # (without breaking below it).
            if np.isfinite(min_low_dist):
                # min_low_dist is >=0; good if <= ~touch_tol
                min_touch_score = _st_score_smaller_better(min_low_dist, good=good_dist, bad=bad_dist)
            else:
                min_touch_score = 0.4

            # Touch-count score: more touches => more validated support (up to a point)
            touches_score = _st_score_larger_better(float(touches), bad=0.0, good=3.0)
            if touches >= 6:
                touches_score *= 0.9

            # Approach quality: mild pullback into support is preferable to panic
            look = min(30, len(tail))
            recent_peak = float(close.iloc[-look:].max())
            pullback = (recent_peak - close_last) / recent_peak if recent_peak > 0 else float("nan")
            approach_score = _st_score_in_band(pullback, good_lo=0.03, good_hi=0.14, bad_lo=0.00, bad_hi=0.28) if np.isfinite(pullback) else 0.5

            # Reversal-candle bonus (lightweight): last bar touched near support and closed strong
            rng = max(1e-12, high_last - low_last)
            close_pos = (close_last - low_last) / rng
            touched_now = (low_last >= line_now) and ((low_last - line_now) / line_now <= touch_tol)
            bullish_body = close_last > open_last
            upper_wick = (high_last - max(open_last, close_last)) / rng
            lower_wick = (min(open_last, close_last) - low_last) / rng

            if touched_now:
                # prefer: bullish close, close in upper half, not huge upper wick; some lower wick ok
                reversal_score = (
                    0.35 * (1.0 if bullish_body else 0.0)
                    + 0.40 * _st_score_larger_better(close_pos, bad=0.35, good=0.70)
                    + 0.15 * _st_score_smaller_better(upper_wick, good=0.15, bad=0.55)
                    + 0.10 * _st_score_larger_better(lower_wick, bad=0.05, good=0.35)
                )
            else:
                reversal_score = 0.15  # neutral-low if not touching intrabar

            # Optional: slope sanity (prefer not strongly negative)
            slope_pct_per_bar = m / line_now
            slope_score = _st_score_in_band(slope_pct_per_bar, good_lo=-0.0002, good_hi=0.0012, bad_lo=-0.0025, bad_hi=0.0035)

            # Final score (weights emphasize your constraints + proximity)
            total = float(
                np.average(
                    [
                        proximity_score,
                        min_touch_score,
                        spacing_score,
                        duration_score,
                        touches_score,
                        pivot_sig,
                        approach_score,
                        reversal_score,
                        slope_score,
                    ],
                    weights=[0.26, 0.12, 0.18, 0.10, 0.08, 0.12, 0.06, 0.05, 0.03],
                )
            )
            total = _clamp(total)

            cand = {
                "score": float(total),
                "zz_thr": float(zz_thr),
                "tol": float(tol),
                "good_dist": float(good_dist),
                "bad_dist": float(bad_dist),
                "touch_tol": float(touch_tol),
                "i_low1": int(i1),
                "i_low2": int(i2),
                "low1": float(p1),
                "low2": float(p2),
                "d12": int(d12),
                "d2t": int(d2t),
                "spacing_ratio": float(spacing_ratio),
                "slope_m": float(m),
                "intercept_c": float(c),
                "line_now": float(line_now),
                "dist_above": float(dist_above),
                "touches_after_p2": int(touches),
                "min_low_dist_after_p2": float(min_low_dist) if np.isfinite(min_low_dist) else None,
                "bounce1": float(bounce1) if np.isfinite(bounce1) else None,
                "bounce2": float(bounce2) if np.isfinite(bounce2) else None,
                "pullback_30": float(pullback) if np.isfinite(pullback) else None,
                "touched_now": bool(touched_now),
                "components": {
                    "proximity": float(proximity_score),
                    "min_touch": float(min_touch_score),
                    "spacing": float(spacing_score),
                    "duration": float(duration_score),
                    "touches": float(touches_score),
                    "pivot_sig": float(pivot_sig),
                    "approach": float(approach_score),
                    "reversal": float(reversal_score),
                    "slope": float(slope_score),
                },
            }

            if "date" in tail.columns:
                try:
                    cand["date_low1"] = str(tail["date"].iloc[i1])
                    cand["date_low2"] = str(tail["date"].iloc[i2])
                    cand["date_asof"] = str(tail["date"].iloc[end_idx])
                except Exception:
                    pass

            if best is None or float(cand["score"]) > float(best["score"]):
                best = cand

    if best is None:
        return None

    # Gating: require genuinely "near support" and a minimally valid structure
    MIN_SCORE = 0.40
    if float(best["score"]) < MIN_SCORE or float(best["components"]["proximity"]) < 0.35:
        return None

    ticker, asof = _last_row_info(df)
    return Signal(
        name="support_test",
        ticker=ticker,
        asof=asof,
        score=round(float(best["score"]), 4),
        details=best,
    )


### RESISTANCE 4TH TEST START ##########################################


def _r4_score_smaller_better(x: float, good: float, bad: float) -> float:
    """1 when x <= good, 0 when x >= bad, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if good >= bad:
        return 1.0 if x <= good else 0.0
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return float((bad - x) / (bad - good))

def _r4_score_larger_better(x: float, bad: float, good: float) -> float:
    """0 when x <= bad, 1 when x >= good, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if bad >= good:
        return 1.0 if x >= good else 0.0
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return float((x - bad) / (good - bad))


def _r4_finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")



@register_signal("4th_resistance_test", required_features=[])
def fourth_resistance_test(df: pd.DataFrame) -> Signal | None:
    """
    4th resistance test detector (revised):

    - Resistance line from 3 significant swing-high pivots (long-term).
    - HARD RULE: no candle HIGH may cross above the resistance line between
      pivot1->pivot2, pivot2->pivot3, and pivot3->(today-1), beyond a small tolerance.
      If it does, the higher point should have been a pivot -> reject candidate.
    - Prefer flat or downward sloping resistance; steep upward lines penalized heavily
      (and extremely steep upward lines rejected).
    - Prefer longer-lived lines; score ramps up to ~1 year from first pivot to now.

    Returns Signal(score in [0,1]) or None.
    """
    if df is None or df.empty or len(df) < 140:
        return None

    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date")
    d = d.reset_index(drop=True)

    MAX_LOOKBACK = 520  # ~2 years of daily bars
    tail = d.iloc[-min(len(d), MAX_LOOKBACK):].copy().reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in tail.columns:
            tail[col] = pd.to_numeric(tail[col], errors="coerce")
    if tail[["high", "low", "close", "volume"]].isna().any().any():
        return None

    high = tail["high"]
    low = tail["low"]
    close = tail["close"]
    volume = tail["volume"]

    end_idx = len(tail) - 1
    close_last = float(close.iloc[-1])
    high_last = float(high.iloc[-1])
    vol_last = float(volume.iloc[-1])  # ensure defined

    # Volatility-scaled thresholds
    atr14 = _atr(high, low, close, n=14)
    natr14 = atr14 / close.replace(0, np.nan)

    recent_natr = natr14.iloc[-min(180, len(natr14)) :].to_numpy(dtype=float)
    natr_med = _r4_finite_median(recent_natr[(recent_natr > 0) & np.isfinite(recent_natr)])

    zz_thr = 0.05 if not np.isfinite(natr_med) else float(np.clip(1.6 * natr_med, 0.03, 0.14))
    tol = 0.012 if not np.isfinite(natr_med) else float(np.clip(1.2 * natr_med, 0.006, 0.05))

    # Pivot fit tolerance (line must pass close to the 3 pivots)
    pivot_tol = float(max(0.70 * tol, 0.006))

    # "No-cross" tolerance for highs between pivots (tight; this enforces your rule #1)
    cross_tol = float(np.clip(0.28 * tol, 0.0015, 0.015))

    # Disallow early breakouts via closes above line before today
    close_break_tol = float(max(0.55 * tol, 0.006))

    # Current proximity thresholds (relative to line_now)
    good_dist = float(np.clip(0.18 * tol, 0.0015, 0.010))
    bad_dist = float(np.clip(1.90 * tol, 0.012, 0.060))

    # Volume context
    vma10 = volume.rolling(10, min_periods=10).mean()
    vma50 = volume.rolling(50, min_periods=50).mean()
    vma50_last = float(vma50.iloc[-1]) if np.isfinite(float(vma50.iloc[-1])) else float("nan")

    # Pivot detection window
    ANALYSIS_WIN = int(min(len(tail), 360))
    w_start = len(tail) - ANALYSIS_WIN
    w = tail.iloc[w_start:].reset_index(drop=True)

    # Use highs for resistance pivots
    w_high = w["high"].to_numpy(dtype=float)
    piv = _zigzag_pivots(w_high, threshold=zz_thr)
    piv, typ = _label_pivots(w_high, piv)

    piv_t = [w_start + int(i) for i in piv]
    typ_t = typ[:]
    pivot_highs = [piv_t[i] for i in range(len(piv_t)) if typ_t[i] == "H"]
    if len(pivot_highs) < 3:
        return None

    # Constraints
    MIN_AGE_BARS = 42          # ~2 months (hard requirement)
    MIN_PIVOT_GAP = 8
    MIN_AFTER_THIRD = 3
    MAX_AFTER_THIRD = 180

    # Limit candidate search
    pivot_highs = pivot_highs[-30:]
    idx = np.arange(len(tail), dtype=float)
    best: dict | None = None

    def _seg_max_exceed(seg: np.ndarray, line_arr: np.ndarray) -> float:
        """Max (high-line)/line over seg. Returns -inf if empty."""
        if seg.size == 0:
            return float("-inf")
        hs = high.iloc[seg].to_numpy(dtype=float)
        ln = line_arr[seg]
        ex = (hs - ln) / np.maximum(ln, 1e-12)
        return float(np.nanmax(ex))

    for a in range(len(pivot_highs) - 2):
        i1 = int(pivot_highs[a])
        age_bars = end_idx - i1
        if age_bars < MIN_AGE_BARS:
            continue

        for b in range(a + 1, len(pivot_highs) - 1):
            i2 = int(pivot_highs[b])
            if (i2 - i1) < MIN_PIVOT_GAP:
                continue

            for c in range(b + 1, len(pivot_highs)):
                i3 = int(pivot_highs[c])
                if (i3 - i2) < MIN_PIVOT_GAP:
                    continue

                bars_since_third = end_idx - i3
                if bars_since_third < MIN_AFTER_THIRD or bars_since_third > MAX_AFTER_THIRD:
                    continue

                xs = np.array([i1, i2, i3], dtype=float)
                ys = np.array([float(high.iloc[i1]), float(high.iloc[i2]), float(high.iloc[i3])], dtype=float)
                if not np.isfinite(ys).all() or np.any(ys <= 0):
                    continue

                # Fit resistance line to 3 highs
                m, k = np.polyfit(xs, ys, 1)
                line = m * idx + k
                line_now = float(line[end_idx])
                if not np.isfinite(line_now) or line_now <= 0:
                    continue

                # Pivots must be close to the line
                fit_vals = m * xs + k
                errs = np.abs(ys - fit_vals) / np.maximum(fit_vals, 1e-12)
                if np.any(errs > pivot_tol):
                    continue

                # RULE #1: no highs above the line between pivots (excluding pivot candles)
                seg12 = np.arange(i1 + 1, i2, dtype=int)
                seg23 = np.arange(i2 + 1, i3, dtype=int)
                seg3n = np.arange(i3 + 1, end_idx, dtype=int)  # before today

                max_ex_12 = _seg_max_exceed(seg12, line)
                if np.isfinite(max_ex_12) and max_ex_12 > cross_tol:
                    continue
                max_ex_23 = _seg_max_exceed(seg23, line)
                if np.isfinite(max_ex_23) and max_ex_23 > cross_tol:
                    continue
                max_ex_3n = _seg_max_exceed(seg3n, line)
                if np.isfinite(max_ex_3n) and max_ex_3n > cross_tol:
                    continue

                # No closes meaningfully above line prior to today (prevents an earlier breakout)
                seg_all = np.arange(i1, end_idx, dtype=int)
                if seg_all.size < 25:
                    continue
                close_seg = close.iloc[seg_all].to_numpy(dtype=float)
                line_seg = line[seg_all]
                if np.any(close_seg > line_seg * (1.0 + close_break_tol)):
                    continue

                # Strict: exactly 3 pivot-high touches (3 tests already), so current is plausibly the 4th
                near_pivots = []
                for ph in pivot_highs:
                    ph = int(ph)
                    if ph < i1 or ph >= end_idx:
                        continue
                    rel = abs(float(high.iloc[ph]) - float(line[ph])) / max(float(line[ph]), 1e-12)
                    if rel <= pivot_tol:
                        near_pivots.append(ph)
                if sorted(near_pivots) != sorted([i1, i2, i3]):
                    continue

                # Current proximity to the line
                touch_dist = min(abs(line_now - close_last), abs(line_now - high_last)) / max(line_now, 1e-12)
                if not np.isfinite(touch_dist) or touch_dist > bad_dist:
                    continue

                breakout = close_last > line_now
                if breakout:
                    # allow only an early breakout (not extended)
                    ext = (close_last / line_now) - 1.0
                    if ext > max(0.06, 3.0 * bad_dist):
                        continue

                # --- Scoring ---

                proximity_score = _r4_score_smaller_better(touch_dist, good=good_dist, bad=bad_dist)

                # RULE #3: age preference up to ~1 year
                age_score = _r4_score_larger_better(float(age_bars), bad=float(MIN_AGE_BARS), good=252.0)

                # Pivot distribution across lifetime: prefer roughly 1/3 and 2/3 placements
                L = max(1.0, float(end_idx - i1))
                t2 = float(i2 - i1) / L
                t3 = float(i3 - i1) / L
                dist_sum = abs(t2 - (1.0 / 3.0)) + abs(t3 - (2.0 / 3.0))
                distribution_score = _r4_score_smaller_better(dist_sum, good=0.12, bad=0.46)

                mean_err = float(np.mean(errs))
                pivot_fit_score = _r4_score_smaller_better(mean_err, good=0.0015, bad=pivot_tol)

                # Respect: with rule #1, max exceed is bounded; best when near zero
                max_ex_all = max(
                    max_ex_12 if np.isfinite(max_ex_12) else -np.inf,
                    max_ex_23 if np.isfinite(max_ex_23) else -np.inf,
                    max_ex_3n if np.isfinite(max_ex_3n) else -np.inf,
                )
                respect_score = _r4_score_smaller_better(max_ex_all, good=0.0005, bad=cross_tol)

                # RULE #2: slope preference: flat/down best; steep up worst
                slope_per_bar = float(m / line_now)  # ~percent per bar
                if slope_per_bar <= 0:
                    slope_score = _r4_score_smaller_better(abs(slope_per_bar), good=0.00005, bad=0.0020)
                else:
                    slope_score = _r4_score_smaller_better(slope_per_bar, good=0.00003, bad=0.00085)

                # extra penalty / reject if too steep upward
                if slope_per_bar > 0.0025:
                    continue
                slope_penalty = 1.0
                if slope_per_bar > 0.0015:
                    slope_penalty = 0.55
                elif slope_per_bar > 0.00085:
                    slope_penalty = 0.75

                # Timing since 3rd test (avoid too immediate or stale)
                if bars_since_third <= 10:
                    third_timing = float(bars_since_third / 10.0)
                elif bars_since_third <= 70:
                    third_timing = 1.0
                elif bars_since_third <= 140:
                    third_timing = float(1.0 - 0.6 * ((bars_since_third - 70) / 70.0))
                else:
                    third_timing = 0.4
                third_timing_score = _clamp(third_timing)

                # Rising lows / pressure build (last ~60 bars)
                Np = min(60, len(tail))
                if Np >= 20:
                    y_low = low.iloc[-Np:].to_numpy(dtype=float)
                    x = np.arange(Np, dtype=float)
                    x0 = x - x.mean()
                    y0 = y_low - y_low.mean()
                    denom = float(np.dot(x0, x0))
                    slope_low = float(np.dot(x0, y0) / denom) if denom > 0 else 0.0
                    slope_low_norm = slope_low / max(1e-12, float(np.mean(y_low)))
                else:
                    slope_low_norm = float("nan")
                rising_lows_score = _r4_score_larger_better(slope_low_norm, bad=-0.0012, good=0.0018)

                # Volatility contraction (ATR/close ratio)
                natr_arr = natr14.to_numpy(dtype=float)
                if len(natr_arr) >= 70:
                    late = _r4_finite_median(natr_arr[-20:])
                    early = _r4_finite_median(natr_arr[-70:-30])
                    natr_ratio = (late / max(early, 1e-12)) if np.isfinite(late) and np.isfinite(early) else float("nan")
                else:
                    natr_ratio = float("nan")
                vol_contraction_score = _r4_score_smaller_better(natr_ratio, good=0.72, bad=1.15) if np.isfinite(natr_ratio) else 0.5

                # Volume dry-up into test
                v10_last = float(vma10.iloc[-1]) if np.isfinite(float(vma10.iloc[-1])) else float("nan")
                if np.isfinite(v10_last) and np.isfinite(vma50_last) and vma50_last > 0:
                    vol_dry_ratio = v10_last / vma50_last
                    vol_dry_score = _r4_score_smaller_better(vol_dry_ratio, good=0.70, bad=1.20)
                else:
                    vol_dry_ratio = None
                    vol_dry_score = 0.5

                # Breakout readiness / breakout quality
                if breakout and np.isfinite(vma50_last) and vma50_last > 0:
                    ext = (close_last / line_now) - 1.0
                    ext_score = _r4_score_smaller_better(ext, good=0.01, bad=0.12)
                    vol_part = vol_last / vma50_last
                    vol_part_score = _r4_score_larger_better(vol_part, bad=0.90, good=1.70)
                    breakout_score = ext_score * vol_part_score
                else:
                    dist_under = (line_now - close_last) / line_now if close_last <= line_now else 0.0
                    breakout_score = _r4_score_smaller_better(dist_under, good=good_dist, bad=bad_dist)

                total = float(
                    np.average(
                        [
                            proximity_score,
                            distribution_score,
                            age_score,
                            pivot_fit_score,
                            respect_score,
                            third_timing_score,
                            slope_score,
                            rising_lows_score,
                            vol_contraction_score,
                            vol_dry_score,
                            breakout_score,
                        ],
                        # Age and slope weights bumped vs prior versions to reflect your priorities.
                        weights=[0.23, 0.12, 0.14, 0.08, 0.11, 0.05, 0.07, 0.07, 0.05, 0.04, 0.04],
                    )
                )
                total = _clamp(total * slope_penalty)

                cand = {
                    "score": float(total),
                    "zz_thr": float(zz_thr),
                    "tol": float(tol),
                    "pivot_tol": float(pivot_tol),
                    "cross_tol": float(cross_tol),
                    "close_break_tol": float(close_break_tol),
                    "i_pivots": [int(i1), int(i2), int(i3)],
                    "pivot_highs": [float(high.iloc[i1]), float(high.iloc[i2]), float(high.iloc[i3])],
                    "age_bars": int(age_bars),
                    "bars_since_third": int(bars_since_third),
                    "m": float(m),
                    "k": float(k),
                    "slope_per_bar": float(slope_per_bar),
                    "line_now": float(line_now),
                    "touch_dist": float(touch_dist),
                    "breakout": bool(breakout),
                    "max_exceed": {"12": float(max_ex_12), "23": float(max_ex_23), "3n": float(max_ex_3n)},
                    "vol_dry_ratio": float(vol_dry_ratio) if vol_dry_ratio is not None else None,
                    "natr_ratio": float(natr_ratio) if np.isfinite(natr_ratio) else None,
                    "distribution": {"t2": float(t2), "t3": float(t3), "dist_sum": float(dist_sum)},
                    "pivot_fit": {"errs": [float(e) for e in errs.tolist()], "mean_err": float(mean_err)},
                    "components": {
                        "proximity": float(proximity_score),
                        "distribution": float(distribution_score),
                        "age": float(age_score),
                        "pivot_fit": float(pivot_fit_score),
                        "respect": float(respect_score),
                        "third_timing": float(third_timing_score),
                        "slope": float(slope_score),
                        "slope_penalty": float(slope_penalty),
                        "rising_lows": float(rising_lows_score),
                        "vol_contraction": float(vol_contraction_score),
                        "vol_dry": float(vol_dry_score),
                        "breakout": float(breakout_score),
                    },
                }

                if "date" in tail.columns:
                    try:
                        cand["date_pivots"] = [
                            str(tail["date"].iloc[i1]),
                            str(tail["date"].iloc[i2]),
                            str(tail["date"].iloc[i3]),
                        ]
                        cand["date_asof"] = str(tail["date"].iloc[end_idx])
                    except Exception:
                        pass

                if best is None or float(cand["score"]) > float(best["score"]):
                    best = cand

    if best is None:
        return None

    MIN_SCORE = 0.42
    if float(best["score"]) < MIN_SCORE or float(best["components"]["proximity"]) < 0.35:
        return None

    ticker, asof = _last_row_info(df)
    return Signal(
        name="4th_resistance_test",
        ticker=ticker,
        asof=asof,
        score=round(float(best["score"]), 4),
        details=best,
    )


### WAVE 2 FINDING ###################################
def _w2_score_smaller_better(x: float, good: float, bad: float) -> float:
    """1 when x <= good, 0 when x >= bad, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if good >= bad:
        return 1.0 if x <= good else 0.0
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return float((bad - x) / (bad - good))

def _w2_score_larger_better(x: float, bad: float, good: float) -> float:
    """0 when x <= bad, 1 when x >= good, linear in between."""
    if not np.isfinite(x):
        return 0.0
    if bad >= good:
        return 1.0 if x >= good else 0.0
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return float((x - bad) / (good - bad))

def _w2_score_band(x: float, best_lo: float, best_hi: float, bad_lo: float, bad_hi: float) -> float:
    """
    1 inside [best_lo, best_hi], 0 outside [bad_lo, bad_hi], linear ramps between.
    """
    if not np.isfinite(x):
        return 0.0
    if x <= bad_lo or x >= bad_hi:
        return 0.0
    if best_lo <= x <= best_hi:
        return 1.0
    if x < best_lo:
        return float((x - bad_lo) / max(1e-12, (best_lo - bad_lo)))
    return float((bad_hi - x) / max(1e-12, (bad_hi - best_hi)))

def _w2_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _w2_finite_median(arr) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


@register_signal("wave2", required_features=[])
def wave2(df: pd.DataFrame) -> Signal | None:
    """
    Elliott Wave "Wave 2 pivot" setup:
      Wave 1: swing low -> swing high (impulsive advance)
      Wave 2: retrace down but DOES NOT go below Wave 1 start
      Now: near Wave 2 low and turning up (pivot forming) -> best Wave 3 entry posture

    Input: OHLCV. Output: Signal(score in [0,1]) or None.
    """
    if df is None or df.empty or len(df) < 140:
        return None

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date")
    d = d.reset_index(drop=True)

    # Lookback
    MAX_LOOKBACK = 520
    tail = d.iloc[-min(len(d), MAX_LOOKBACK):].copy().reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        tail[col] = pd.to_numeric(tail[col], errors="coerce")

    if tail[["open", "high", "low", "close", "volume"]].isna().any().any():
        return None

    o = tail["open"]
    h = tail["high"]
    l = tail["low"]
    c = tail["close"]
    v = tail["volume"]

    end_idx = len(tail) - 1
    close_last = float(c.iloc[-1])
    open_last = float(o.iloc[-1])
    high_last = float(h.iloc[-1])
    low_last = float(l.iloc[-1])
    vol_last = float(v.iloc[-1])

    # Volatility -> zigzag threshold + tolerances
    atr14 = _w2_atr(h, l, c, n=14)
    natr14 = atr14 / c.replace(0, np.nan)

    recent_natr = natr14.iloc[-min(120, len(natr14)) :].to_numpy(dtype=float)
    natr_med = _w2_finite_median(recent_natr[(recent_natr > 0) & np.isfinite(recent_natr)])

    zz_thr = 0.05 if not np.isfinite(natr_med) else float(np.clip(1.6 * natr_med, 0.03, 0.15))
    tol = 0.012 if not np.isfinite(natr_med) else float(np.clip(1.2 * natr_med, 0.006, 0.06))

    # “Wave2 low should hold” tolerance (post-low undercut)
    undercut_tol = float(np.clip(0.35 * tol, 0.002, 0.02))

    # Proximity to wave2 low: entry shouldn’t be too extended
    max_entry_ext = float(max(0.15, 6.0 * tol))  # > this is likely already wave3 extended

    # Pivots via your existing functions
    prices = c.to_numpy(dtype=float)
    piv = _zigzag_pivots(prices, threshold=zz_thr)
    piv, typ = _label_pivots(prices, piv)

    if len(piv) < 5:
        return None

    piv = [int(x) for x in piv]
    typ = [str(t) for t in typ]

    # Candidate constraints
    L2_MAX_AGE = 45          # wave2 low must be relatively recent to be “turning now”
    MAX_W1_LEN = 220
    MAX_W2_LEN = 220
    MIN_W1_LEN = 6
    MIN_W2_LEN = 6

    best: dict | None = None

    # Precompute rolling aids
    sma5 = c.rolling(5, min_periods=5).mean()
    sma5_last = float(sma5.iloc[-1]) if np.isfinite(float(sma5.iloc[-1])) else float("nan")
    vma50 = v.rolling(50, min_periods=50).mean()
    vma50_last = float(vma50.iloc[-1]) if np.isfinite(float(vma50.iloc[-1])) else float("nan")

    # Enumerate L0-H1-L2 where L2 is a pivot low near the end.
    for i in range(len(piv)):
        if typ[i] != "L":
            continue
        i0 = piv[i]

        for k in range(i + 1, len(piv)):
            if typ[k] != "H":
                continue
            i1 = piv[k]

            # Wave1 length constraints
            w1_len = i1 - i0
            if w1_len < MIN_W1_LEN or w1_len > MAX_W1_LEN:
                continue

            low0 = float(l.iloc[i0])
            high1 = float(h.iloc[i1])
            if not (np.isfinite(low0) and np.isfinite(high1)) or low0 <= 0 or high1 <= 0:
                continue
            if high1 <= low0:
                continue

            # Wave1 should be "clean": no lower low below start inside wave1 (avoid mis-anchoring)
            wave1_min_low = float(l.iloc[i0 : i1 + 1].min())
            if wave1_min_low < low0 * (1.0 - undercut_tol):
                continue

            # Impulse size
            up1 = (high1 - low0) / low0
            if up1 < max(0.06, 1.1 * zz_thr):
                continue

            # Now choose wave2 low pivot after wave1 high
            for j in range(k + 1, len(piv)):
                if typ[j] != "L":
                    continue
                i2 = piv[j]

                w2_len = i2 - i1
                if w2_len < MIN_W2_LEN or w2_len > MAX_W2_LEN:
                    continue

                age2 = end_idx - i2
                if age2 < 0 or age2 > L2_MAX_AGE:
                    continue

                low2 = float(l.iloc[i2])
                if not np.isfinite(low2) or low2 <= 0:
                    continue

                # HARD Elliott constraint: wave2 must not break wave1 start
                if low2 < low0 * (1.0 - undercut_tol):
                    continue

                # Ensure wave1 top is actually the top before wave2 bottom:
                # no higher high after i1 before i2 (otherwise wave1 top is later)
                hi_after = float(h.iloc[i1 : i2 + 1].max())
                if hi_after > high1 * (1.0 + max(0.6 * tol, 0.006)):
                    continue

                # wave2 low should be near the bottom of the correction segment
                min_low_w2 = float(l.iloc[i1 : i2 + 1].min())
                if low2 > min_low_w2 * (1.0 + max(0.8 * tol, 0.01)):
                    continue

                # Post-low integrity: after i2, no new low below low2 (else not “turning”)
                post_min = float(l.iloc[i2 : end_idx + 1].min())
                if post_min < low2 * (1.0 - undercut_tol):
                    continue

                # Entry extension: current close not too far above low2
                ext_from_low2 = (close_last - low2) / low2
                if ext_from_low2 < -0.002:
                    continue
                if ext_from_low2 > max_entry_ext:
                    continue

                # Retracement (Fib-ish preference)
                retr = (high1 - low2) / max(1e-12, (high1 - low0))
                # practical rejection: not too shallow, not too deep
                if retr < 0.18 or retr > 0.95:
                    continue

                # Time ratio preference (wave2 often ~0.6–2.5x wave1)
                ratio = (w2_len / max(1, w1_len))
                time_ratio = max(ratio, 1.0 / max(ratio, 1e-12))

                # Retest bonus: detect an earlier low in wave2 close to low2 with a bounce between
                retest_score = 0.35
                prev_lows = [piv[t] for t in range(k + 1, j) if typ[t] == "L"]
                if prev_lows:
                    prev = int(prev_lows[-1])
                    if (i2 - prev) >= 5:
                        low_prev = float(l.iloc[prev])
                        sim = abs(low_prev - low2) / low2
                        # ensure a bounce high pivot exists between prev and i2
                        has_bounce = any((typ[t] == "H" and prev < piv[t] < i2) for t in range(k + 1, j))
                        if has_bounce and sim <= max(0.02, 1.2 * tol):
                            # higher low is best; small undercut acceptable but weaker
                            hl = (low2 / low_prev) - 1.0
                            if hl >= 0:
                                retest_score = 0.95 * _w2_score_smaller_better(sim, good=0.004, bad=max(0.02, 1.2 * tol)) + 0.05
                            else:
                                retest_score = 0.65 * _w2_score_smaller_better(sim, good=0.004, bad=max(0.02, 1.2 * tol)) + 0.05

                # Volume / volatility behavior (light/moderate weight)
                seg_w1 = slice(i0, i1 + 1)
                seg_w2 = slice(i1, i2 + 1)
                v1 = float(v.iloc[seg_w1].mean())
                v2 = float(v.iloc[seg_w2].mean())
                vol_ratio = (v2 / v1) if (np.isfinite(v1) and v1 > 0 and np.isfinite(v2)) else float("nan")
                vol_score = _w2_score_smaller_better(vol_ratio, good=0.70, bad=1.25) if np.isfinite(vol_ratio) else 0.5

                n1 = float(natr14.iloc[seg_w1].mean())
                n2 = float(natr14.iloc[seg_w2].mean())
                natr_ratio = (n2 / n1) if (np.isfinite(n1) and n1 > 0 and np.isfinite(n2)) else float("nan")
                natr_score = _w2_score_smaller_better(natr_ratio, good=0.85, bad=1.35) if np.isfinite(natr_ratio) else 0.5

                # Turning/pivot quality on the right edge
                # Candle anatomy
                rng = max(1e-12, high_last - low_last)
                close_pos = (close_last - low_last) / rng
                bullish = 1.0 if close_last >= open_last else 0.0
                lower_wick = (min(open_last, close_last) - low_last) / rng
                upper_wick = (high_last - max(open_last, close_last)) / rng

                reversal_candle_score = (
                    0.35 * bullish
                    + 0.30 * _w2_score_larger_better(close_pos, bad=0.30, good=0.70)
                    + 0.20 * _w2_score_larger_better(lower_wick, bad=0.05, good=0.35)
                    + 0.15 * _w2_score_smaller_better(upper_wick, good=0.15, bad=0.55)
                )

                # Micro momentum shift: close above prior high OR above SMA5
                prev_high = float(h.iloc[-2]) if len(h) >= 2 else float("nan")
                cond1 = 1.0 if (np.isfinite(prev_high) and close_last > prev_high) else 0.0
                cond2 = _w2_score_larger_better(close_last / sma5_last, bad=0.992, good=1.010) if (np.isfinite(sma5_last) and sma5_last > 0) else 0.5
                micro_break_score = 0.55 * cond1 + 0.45 * cond2

                # Recency of low2: very recent is best for “entry at pivot”
                age_score = _w2_score_smaller_better(float(age2), good=2.0, bad=28.0)

                # Entry distance from low2: best slightly off the low but not extended
                dist_score = _w2_score_band(ext_from_low2, best_lo=0.008, best_hi=0.065, bad_lo=0.0, bad_hi=max_entry_ext)

                # Structure scores
                wave1_strength = _w2_score_larger_better(up1, bad=0.06, good=0.22)
                retr_score = _w2_score_band(retr, best_lo=0.45, best_hi=0.70, bad_lo=0.20, bad_hi=0.92)
                above_start_margin = (low2 / low0) - 1.0
                above_start_score = _w2_score_larger_better(above_start_margin, bad=0.0, good=0.08)

                time_score = _w2_score_smaller_better(time_ratio, good=1.40, bad=3.50)

                structure_score = float(
                    np.average(
                        [retr_score, above_start_score, time_score, retest_score],
                        weights=[0.38, 0.22, 0.14, 0.26],
                    )
                )

                turning_score = float(
                    np.average(
                        [age_score, dist_score, reversal_candle_score, micro_break_score, natr_score],
                        weights=[0.24, 0.26, 0.24, 0.16, 0.10],
                    )
                )

                behavior_score = float(
                    np.average([vol_score, natr_score], weights=[0.55, 0.45])
                )

                total = float(
                    np.average(
                        [wave1_strength, structure_score, turning_score, behavior_score],
                        weights=[0.22, 0.33, 0.35, 0.10],
                    )
                )
                total = float(_clamp(total, 0.0, 1.0))

                cand = {
                    "score": float(total),
                    "zz_thr": float(zz_thr),
                    "tol": float(tol),
                    "i_wave1_start": int(i0),
                    "i_wave1_end": int(i1),
                    "i_wave2_low": int(i2),
                    "wave1_start_low": float(low0),
                    "wave1_end_high": float(high1),
                    "wave2_low": float(low2),
                    "wave1_return": float(up1),
                    "wave2_retracement": float(retr),
                    "wave2_age_bars": int(age2),
                    "wave1_len": int(w1_len),
                    "wave2_len": int(w2_len),
                    "time_ratio_sym": float(time_ratio),
                    "ext_from_wave2_low": float(ext_from_low2),
                    "vol_ratio_w2_w1": float(vol_ratio) if np.isfinite(vol_ratio) else None,
                    "natr_ratio_w2_w1": float(natr_ratio) if np.isfinite(natr_ratio) else None,
                    "retest_bonus": float(retest_score),
                    "components": {
                        "wave1_strength": float(wave1_strength),
                        "structure": float(structure_score),
                        "turning": float(turning_score),
                        "behavior": float(behavior_score),
                        "retr": float(retr_score),
                        "above_start": float(above_start_score),
                        "time": float(time_score),
                        "reversal_candle": float(reversal_candle_score),
                        "micro_break": float(micro_break_score),
                        "age": float(age_score),
                        "dist": float(dist_score),
                        "vol": float(vol_score),
                        "natr": float(natr_score),
                    },
                }

                if "date" in tail.columns:
                    try:
                        cand["date_wave1_start"] = str(tail["date"].iloc[i0])
                        cand["date_wave1_end"] = str(tail["date"].iloc[i1])
                        cand["date_wave2_low"] = str(tail["date"].iloc[i2])
                        cand["date_asof"] = str(tail["date"].iloc[end_idx])
                    except Exception:
                        pass

                if best is None or float(cand["score"]) > float(best["score"]):
                    best = cand

    if best is None:
        return None

    # Gate: require both structure and turning quality (prevents “random retracement” matches)
    MIN_SCORE = 0.42
    if float(best["score"]) < MIN_SCORE:
        return None
    if float(best["components"]["structure"]) < 0.38 or float(best["components"]["turning"]) < 0.34:
        return None

    ticker, asof = _last_row_info(df)
    return Signal(
        name="wave2",
        ticker=ticker,
        asof=asof,
        score=round(float(best["score"]), 4),
        details=best,
    )