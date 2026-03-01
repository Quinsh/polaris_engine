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

def _vcp_clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
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

def _vcp_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _vcp_zigzag_pivots(prices: np.ndarray, threshold: float) -> list[int]:
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

def _vcp_label_pivots(prices: np.ndarray, pivots: list[int]) -> tuple[list[int], list[str]]:
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
    atr14 = _vcp_atr(high, low, close, n=14)
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

    piv = _vcp_zigzag_pivots(w_close, threshold=zz_thr)
    piv, piv_type = _vcp_label_pivots(w_close, piv)

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

        total_score = _vcp_clamp(total_score, 0.0, 1.0)

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

def _db_clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(lo if x < lo else hi if x > hi else x)

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

def _db_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _db_finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")

def _db_zigzag_pivots(prices: np.ndarray, threshold: float) -> list[int]:
    """
    Percent zigzag pivots on a 1D price array.
    threshold is a fraction (e.g. 0.05 for 5%).
    Returns pivot indices (chronological, de-duplicated).
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
            if hi_i > lo_i and hi > 0 and (hi - p) / hi >= thr:
                pivots.append(hi_i)
                trend = -1
                lo = p
                lo_i = i
                hi = p
                hi_i = i
            elif lo_i > hi_i and lo > 0 and (p - lo) / lo >= thr:
                pivots.append(lo_i)
                trend = 1
                hi = p
                hi_i = i
                lo = p
                lo_i = i
        elif trend == 1:
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

    # Add last extreme so the most recent leg is represented
    if trend == 1:
        pivots.append(hi_i)
    elif trend == -1:
        pivots.append(lo_i)
    else:
        pivots = [0, n - 1]

    # Clean increasing + ensure endpoints
    cleaned: list[int] = []
    last = -1
    for idx in pivots:
        idx = int(idx)
        if idx <= last:
            continue
        cleaned.append(idx)
        last = idx

    if not cleaned:
        cleaned = [0, n - 1]
    if cleaned[0] != 0:
        cleaned.insert(0, 0)
    if cleaned[-1] != n - 1:
        cleaned.append(n - 1)

    return cleaned

def _db_label_pivots(prices: np.ndarray, pivots: list[int]) -> tuple[list[int], list[str]]:
    """Label pivots as H/L by neighbor comparison and enforce alternation."""
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

    # Merge consecutive same-type pivots (keep most extreme)
    out_idx: list[int] = [pivots[0]]
    out_typ: list[str] = [types[0]]
    for j in range(1, len(pivots)):
        idx = int(pivots[j])
        typ = types[j]
        if typ == out_typ[-1]:
            prev = out_idx[-1]
            if typ == "H":
                if prices[idx] >= prices[prev]:
                    out_idx[-1] = idx
            else:
                if prices[idx] <= prices[prev]:
                    out_idx[-1] = idx
        else:
            out_idx.append(idx)
            out_typ.append(typ)

    return out_idx, out_typ


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
    atr14 = _db_atr(high, low, close, n=14)
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
    piv = _db_zigzag_pivots(w_close, threshold=zz_thr)
    piv, typ = _db_label_pivots(w_close, piv)

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

        total_score = _db_clamp(total_score, 0.0, 1.0)

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


### SUPPORT TEST
import numpy as np
import pandas as pd

def _st_clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(lo if x < lo else hi if x > hi else x)

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

def _st_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _st_finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")

def _st_zigzag_pivots(prices: np.ndarray, threshold: float) -> list[int]:
    """
    Percent zigzag pivots on a 1D price array.
    threshold is a fraction (e.g. 0.05 for 5%).
    Returns pivot indices (chronological, de-duplicated, includes endpoints).
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
            if hi_i > lo_i and hi > 0 and (hi - p) / hi >= thr:
                pivots.append(hi_i)
                trend = -1
                lo = p
                lo_i = i
                hi = p
                hi_i = i
            elif lo_i > hi_i and lo > 0 and (p - lo) / lo >= thr:
                pivots.append(lo_i)
                trend = 1
                hi = p
                hi_i = i
                lo = p
                lo_i = i
        elif trend == 1:
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

    # add last extreme
    if trend == 1:
        pivots.append(hi_i)
    elif trend == -1:
        pivots.append(lo_i)
    else:
        pivots = [0, n - 1]

    # clean increasing + endpoints
    cleaned: list[int] = []
    last = -1
    for idx in pivots:
        idx = int(idx)
        if idx <= last:
            continue
        cleaned.append(idx)
        last = idx

    if not cleaned:
        cleaned = [0, n - 1]
    if cleaned[0] != 0:
        cleaned.insert(0, 0)
    if cleaned[-1] != n - 1:
        cleaned.append(n - 1)
    return cleaned

def _st_label_pivots(prices: np.ndarray, pivots: list[int]) -> tuple[list[int], list[str]]:
    """Label pivots as H/L by neighbor comparison, then enforce alternation by merging duplicates."""
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

    # merge consecutive same-type pivots (keep most extreme)
    out_idx: list[int] = [pivots[0]]
    out_typ: list[str] = [types[0]]
    for j in range(1, len(pivots)):
        idx = int(pivots[j])
        typ = types[j]
        if typ == out_typ[-1]:
            prev = out_idx[-1]
            if typ == "H":
                if prices[idx] >= prices[prev]:
                    out_idx[-1] = idx
            else:
                if prices[idx] <= prices[prev]:
                    out_idx[-1] = idx
        else:
            out_idx.append(idx)
            out_typ.append(typ)

    return out_idx, out_typ


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
    atr14 = _st_atr(high, low, close, n=14)
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
    piv = _st_zigzag_pivots(w_close, threshold=zz_thr)
    piv, typ = _st_label_pivots(w_close, piv)
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
            total = _st_clamp(total)

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