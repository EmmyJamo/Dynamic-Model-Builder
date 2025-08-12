# Data_Scoring/Voltage/V_P.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from Data_Scoring.Attribute_Detection.Functions import (
    detect_drop_time,
    detect_steady_state,
)

# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

_SCEN_CFG = {
    "fast_dip": dict(
        dip_target=0.90,
        dip_band=(0.88, 0.92),   # acceptable min band
        hold_target=0.20,        # seconds
        hold_tol=0.05,           # ± seconds allowed deviation
        fall_ref=0.10,           # boundary fall time if outside envelope
        rise_max=0.60,           # max allowed recovery time budget
    ),
    "slow_hold": dict(
        dip_target=0.90,
        dip_band=(0.88, 0.92),
        hold_target=5.00,
        hold_tol=0.25,
        fall_ref=0.10,
        rise_max=0.60,           # adjust if spec allows slower
    ),
}

_TIME_COL = "All calculations"   # PF time column name


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _infer_scenario(csv_path_part: Path) -> str:
    s = str(csv_path_part).lower()
    if "slow_hold" in s:
        return "slow_hold"
    if "fast_dip" in s:
        return "fast_dip"
    return "fast_dip"

def _nearest_boundary(val: float, lo: float, hi: float) -> float:
    if val < lo: return lo
    if val > hi: return hi
    return val

def _first_rising_after_min(y: np.ndarray, t: np.ndarray, i_min: int,
                            slope_eps: float = 2e-4,
                            window: int = 5) -> Optional[int]:
    if i_min >= len(y) - 2:
        return None
    dy = np.diff(y) / np.diff(t)
    dy = np.concatenate([dy[:1], dy])  # align length to y
    if window > 1:
        ker = np.ones(window) / window
        dy_s = np.convolve(dy, ker, mode="same")
    else:
        dy_s = dy
    for i in range(i_min, len(dy_s) - window):
        if np.all(dy_s[i:i+window] > slope_eps):
            return i
    return None

def _build_ideal(time: np.ndarray,
                 drop_t: float,
                 final_v: float,
                 cfg: dict,
                 *,
                 in_envelope: bool,
                 min_t: float,
                 min_v: float,
                 hold_start_t: Optional[float],
                 steady_t: Optional[float]) -> np.ndarray:
    dip_lo, dip_hi = cfg["dip_band"]
    hold_target    = cfg["hold_target"]
    hold_tol       = cfg["hold_tol"]
    fall_ref       = cfg["fall_ref"]
    rise_max       = cfg["rise_max"]

    ideal = np.full_like(time, np.nan, dtype=float)
    ideal[time <= drop_t] = 1.0  # pre-drop flat at 1.0 p.u.

    if in_envelope:
        # match measured fall slope and depth
        t_fall_start = drop_t
        t_fall_end   = max(min_t, drop_t + 1e-3)
        v_fall_end   = min_v

        mask_fall = (time >= t_fall_start) & (time <= t_fall_end)
        ideal[mask_fall] = np.interp(time[mask_fall],
                                     [t_fall_start, t_fall_end],
                                     [1.0, v_fall_end])

        # hold at min_v until we detect a rise (or zero-length hold if unknown)
        t_hold_end = max(hold_start_t or min_t, min_t)
        mask_hold  = (time > min_t) & (time <= t_hold_end)
        ideal[mask_hold] = v_fall_end

        # recovery: ramp to final_v by steady_t (or within rise_max if steady_t unknown)
        t_rec_start = t_hold_end
        t_rec_end   = max((steady_t or (t_rec_start + rise_max)), t_rec_start + 1e-3)
        mask_rec    = (time > t_rec_start) & (time <= t_rec_end)
        ideal[mask_rec] = np.interp(time[mask_rec],
                                    [t_rec_start, t_rec_end],
                                    [v_fall_end, final_v])
        ideal[time >= t_rec_end] = final_v

    else:
        # boundary ideal closest to allowed limits
        dip_ref = _nearest_boundary(min_v, dip_lo, dip_hi)

        # fall over reference time
        t_fall_start = drop_t
        t_fall_end   = drop_t + fall_ref
        mask_fall = (time >= t_fall_start) & (time <= t_fall_end)
        ideal[mask_fall] = np.interp(time[mask_fall],
                                     [t_fall_start, t_fall_end],
                                     [1.0, dip_ref])

        # hold for required duration
        t_hold_end = t_fall_end + hold_target
        mask_hold  = (time > t_fall_end) & (time <= t_hold_end)
        ideal[mask_hold] = dip_ref

        # recover within rise_max
        t_rec_end = t_hold_end + rise_max
        mask_rec  = (time > t_hold_end) & (time <= t_rec_end)
        ideal[mask_rec] = np.interp(time[mask_rec],
                                    [t_hold_end, t_rec_end],
                                    [dip_ref, final_v])
        ideal[time >= t_rec_end] = final_v

    # fill any guard-induced NaNs
    if np.isnan(ideal).any():
        s = pd.Series(ideal)
        ideal = s.fillna(method="ffill").fillna(method="bfill").to_numpy()

    return ideal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_voltage_control(busbar: str, csv_path_part) -> float:
    """
    Read '<csv_path_part>/<busbar>.csv', detect drop/hold/recovery,
    create an ideal trace using envelope-aware rules, export the ideal CSV
    next to the sim for debugging, and return a fitness score (MAE with
    optional overshoot penalty).
    """
    csv_root  = Path(csv_path_part)
    scenario  = _infer_scenario(csv_root)
    cfg       = _SCEN_CFG[scenario]

    # load sim
    csv_path = csv_root / f"{busbar}.csv"
    print(csv_path)
    sim_data = pd.read_csv(csv_path, header=0, skiprows=[1])

    time = sim_data[_TIME_COL].to_numpy(dtype=float)
    v    = sim_data[busbar].to_numpy(dtype=float)

    if len(time) < 5 or len(v) != len(time):
        print("⚠️ Not enough points or column mismatch")
        return float("inf")

    # baseline final value from latter half
    final_v = float(np.mean(v[int(len(v)*0.5):]))

    # detect drop point
    drop_idx = detect_drop_time(pd.Series(v))
    if drop_idx is None:
        print("⚠️ No significant drop detected – assuming early drop")
        drop_idx = max(0, int(0.05 * len(time)))
    drop_t = float(time[drop_idx])

    # minimum after drop
    post = v[drop_idx:] if drop_idx < len(v) else v
    min_idx = drop_idx + int(np.argmin(post)) if len(post) else drop_idx
    min_v   = float(v[min_idx])
    min_t   = float(time[min_idx])

    # steady-state timing (from min onwards)
    v_after_min = pd.Series(v[min_idx:])
    ss_rel = detect_steady_state(v_after_min, final_v)
    steady_idx = (min_idx + int(ss_rel)) if ss_rel is not None else None
    steady_t   = float(time[steady_idx]) if steady_idx is not None else None

    # approximate hold end: first sustained rise after min
    hold_start_idx = _first_rising_after_min(v, time, min_idx)
    hold_start_t   = float(time[hold_start_idx]) if hold_start_idx is not None else None

    # envelope checks
    dip_lo, dip_hi = cfg["dip_band"]
    within_dip  = (dip_lo <= min_v <= dip_hi)

    hold_len = (hold_start_t - drop_t) if hold_start_t is not None else 0.0
    within_hold = abs(hold_len - cfg["hold_target"]) <= cfg["hold_tol"]

    if steady_t is not None:
        total_down = steady_t - drop_t
        within_rise = (total_down - max(hold_len, cfg["hold_target"])) <= cfg["rise_max"] + 1e-6
    else:
        within_rise = False

    in_envelope = bool(within_dip and within_hold and within_rise)

    # build ideal
    ideal = _build_ideal(
        time=time,
        drop_t=drop_t,
        final_v=final_v,
        cfg=cfg,
        in_envelope=in_envelope,
        min_t=min_t,
        min_v=min_v,
        hold_start_t=hold_start_t,
        steady_t=steady_t,
    )

    # ----- export ideal for debugging (same folder as sim)
    try:
        ideal_df = pd.DataFrame({
            "All calculations": time,
            f"{busbar}_ideal": ideal,
        })
        ideal_path = csv_root / f"{busbar}_ideal.csv"
        ideal_df.to_csv(ideal_path, index=False)
        print(f"📝 wrote ideal CSV → {ideal_path}")
    except Exception as e:
        print(f"⚠️ ideal export failed: {e}")

    # ----- scoring window
    dt = np.median(np.diff(time))
    t_start = max(time[0], drop_t - 2*dt)
    t_end   = (steady_t + 1.0) if steady_t is not None else time[-1]
    mask    = (time >= t_start) & (time <= t_end)
    if not np.any(mask):
        mask = slice(None)

    mae_voltage  = float(np.mean(np.abs(v[mask] - ideal[mask])))

    vmax_after   = float(np.max(v[min_idx:])) if min_idx < len(v) else float(np.max(v))
    overshoot_pu = max(0.0, (vmax_after - final_v) / max(1e-6, final_v))

    if mae_voltage <= 8.5e-4 and overshoot_pu > 0:
        fitness_value = mae_voltage * (1.0 + overshoot_pu)
    else:
        fitness_value = mae_voltage

    # debug summary
    print(f"Scenario: {scenario}")
    print(f"Drop t = {drop_t:.4f}s, Min v = {min_v:.4f} at t={min_t:.4f}s")
    if steady_t is not None:
        print(f"Steady t = {steady_t:.4f}s; hold≈{hold_len:.3f}s; in_envelope={in_envelope}")
    else:
        print(f"Steady not detected; hold≈{hold_len:.3f}s; in_envelope={in_envelope}")
    print(f"MAE = {mae_voltage:.6g}, Overshoot = {overshoot_pu:.4g}, Fitness = {fitness_value:.6g}")

    return float(fitness_value)
