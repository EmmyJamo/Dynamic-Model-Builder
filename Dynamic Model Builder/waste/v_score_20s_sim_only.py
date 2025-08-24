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
# Scenario configuration (times in seconds, voltages in p.u.)
# Dip targets/bands are specified RELATIVE to V_pre (pre-drop median).
# ---------------------------------------------------------------------------
_SCEN_CFG = {
    "fast_dip": dict(
        dip_target=0.90,          # fraction of V_pre
        dip_band=(0.88, 0.92),    # acceptable fraction band of V_pre
        hold_target=0.20,         # seconds
        hold_tol=0.05,            # ± seconds allowed deviation
        fall_ref=0.10,            # boundary fall time if outside envelope
        rise_max=0.60,            # max allowed recovery budget
    ),
    "slow_hold": dict(
        dip_target=0.90,
        dip_band=(0.88, 0.92),
        hold_target=5.00,
        hold_tol=0.25,
        fall_ref=0.10,
        rise_max=0.60,
    ),
}

# Oscillation & drift heuristics (soft constraints)
_OSC = dict(
    # RMS of high-freq residual allowed in steady state (p.u.)
    rms_ok=0.0015,
    # zero crossings per second allowed (≈ 2×freq). 1.2 ≈ 0.6 Hz nominal.
    zc_per_s_ok=1.2,
    k_rms=2.0,    # strength of RMS penalty
    k_zc=0.25,    # strength of zero-cross penalty
    ma_win_s=0.5, # smoothing window for residual (seconds)
    eval_tail_s=4.0,  # evaluate oscillations on the last N seconds of the run
)

_DRIFT = dict(
    slope_ok=2.0e-4, # |dV/dt| tolerated in p.u./s (we penalise mostly negative)
    bias_ok=0.005,   # allowed late-time bias vs V_pre (p.u.)
    k_slope=3.0,     # slope penalty gain
    k_bias=2.0,      # bias penalty gain
    tail_s=6.0,      # evaluate slope over last N seconds
)

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

def _moving_avg(y: np.ndarray, win_samp: int) -> np.ndarray:
    win = max(1, int(win_samp))
    if win == 1:
        return y.copy()
    ker = np.ones(win, dtype=float) / win
    # total pad = win - 1  → ensures len(output) == len(y) for any parity of win
    left  = win // 2
    right = win - 1 - left
    ypad = np.pad(y, (left, right), mode="edge")
    out = np.convolve(ypad, ker, mode="valid")  # length == len(y)
    return out


def _zero_cross_rate(sig: np.ndarray, t: np.ndarray, amp_eps: float = 3e-4) -> float:
    # count sign changes where amplitude is meaningful
    s = np.sign(sig)
    strong = np.abs(sig) > amp_eps
    zc = 0
    for i in range(1, len(sig)):
        if strong[i-1] and strong[i] and s[i] != s[i-1]:
            zc += 1
    dur = max(1e-6, t[-1] - t[0])
    return zc / dur  # changes per second (~2 × frequency)

def _build_ideal(time: np.ndarray,
                 drop_t: float,
                 final_v: float,
                 cfg: dict,
                 *,
                 in_envelope: bool,
                 min_t: float,
                 min_v: float,
                 hold_start_t: Optional[float],
                 steady_t: Optional[float],
                 V_pre: float) -> np.ndarray:

    # scale dip limits to the actual pre-drop voltage
    dip_lo_rel, dip_hi_rel = cfg["dip_band"]
    dip_lo, dip_hi = dip_lo_rel * V_pre, dip_hi_rel * V_pre

    hold_target    = cfg["hold_target"]
    fall_ref       = cfg["fall_ref"]
    rise_max       = cfg["rise_max"]

    ideal = np.full_like(time, np.nan, dtype=float)
    ideal[time <= drop_t] = V_pre  # pre-drop flat at V_pre p.u.

    if in_envelope:
        # match measured fall slope and depth
        t_fall_start = drop_t
        t_fall_end   = max(min_t, drop_t + 1e-3)
        v_fall_end   = min_v

        mask_fall = (time >= t_fall_start) & (time <= t_fall_end)
        ideal[mask_fall] = np.interp(time[mask_fall],
                                     [t_fall_start, t_fall_end],
                                     [V_pre, v_fall_end])

        # hold at min_v until we detect a rise (or zero-length hold if unknown)
        t_hold_end = max(hold_start_t or min_t, min_t)
        mask_hold  = (time > min_t) & (time <= t_hold_end)
        ideal[mask_hold] = v_fall_end

        # recovery: ramp to final_v by steady_t (or within rise_max if unknown)
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
                                     [V_pre, dip_ref])

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
    penalties for overshoot, sustained oscillations, and steady-state drift).
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

    dt = float(np.median(np.diff(time)))
    T_end = float(time[-1])

    # pre-drop baseline and late-time baseline
    # (use robust medians to ignore spikes)
    V_pre = float(np.median(v[time < (time[0] + max(0.1, 5*dt))]))
    V_late = float(np.median(v[int(len(v)*0.5):]))

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
    ss_rel = detect_steady_state(v_after_min, V_late)
    steady_idx = (min_idx + int(ss_rel)) if ss_rel is not None else None
    steady_t   = float(time[steady_idx]) if steady_idx is not None else None

    # approximate hold end: first sustained rise after min
    hold_start_idx = _first_rising_after_min(v, time, min_idx)
    hold_start_t   = float(time[hold_start_idx]) if hold_start_idx is not None else None

    # envelope checks (scaled to V_pre)
    dip_lo_rel, dip_hi_rel = cfg["dip_band"]
    dip_lo, dip_hi = dip_lo_rel * V_pre, dip_hi_rel * V_pre
    within_dip  = (dip_lo <= min_v <= dip_hi)

    hold_len = (hold_start_t - drop_t) if hold_start_t is not None else 0.0
    within_hold = abs(hold_len - cfg["hold_target"]) <= cfg["hold_tol"]

    if steady_t is not None:
        total_down = steady_t - drop_t
        within_rise = (total_down - max(hold_len, cfg["hold_target"])) <= cfg["rise_max"] + 1e-6
    else:
        within_rise = False

    in_envelope = bool(within_dip and within_hold and within_rise)

    # target final voltage: prefer returning to pre-drop level
    final_v_target = V_pre

    # build ideal (pre-drop segment anchored at V_pre)
    ideal = _build_ideal(
        time=time,
        drop_t=drop_t,
        final_v=final_v_target,
        cfg=cfg,
        in_envelope=in_envelope,
        min_t=min_t,
        min_v=min_v,
        hold_start_t=hold_start_t,
        steady_t=steady_t,
        V_pre=V_pre,
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

    # ----- core MAE on a sensible window (from pre-drop to a bit past settle)
    t_start = max(time[0], drop_t - 2*dt)
    t_end   = (steady_t + 1.0) if steady_t is not None else T_end
    mask    = (time >= t_start) & (time <= t_end)
    if not np.any(mask):
        mask = slice(None)

    mae_voltage  = float(np.mean(np.abs(v[mask] - ideal[mask])))

    # ----- overshoot multiplicative penalty (same as before)
    vmax_after   = float(np.max(v[min_idx:])) if min_idx < len(v) else float(np.max(v))
    overshoot_pu = max(0.0, (vmax_after - final_v_target) / max(1e-6, final_v_target))
    mult = 1.0
    if mae_voltage <= 8.5e-4 and overshoot_pu > 0:
        mult *= (1.0 + overshoot_pu)

    # ----- NEW: oscillation penalty (RMS + zero-cross rate on tail)
    # choose analysis window at the tail (last eval_tail_s seconds)
    t_tail_start = max(t_end, T_end - _OSC["eval_tail_s"])
    tail_mask = time >= t_tail_start
    if np.sum(tail_mask) >= 5:
        win = int(max(1, _OSC["ma_win_s"] / max(dt, 1e-6)))
        baseline = _moving_avg(v, win)
        resid = v - baseline
        resid_tail = resid[tail_mask]
        time_tail  = time[tail_mask]
        rms = float(np.sqrt(np.mean(resid_tail**2)))
        zc_rate = float(_zero_cross_rate(resid_tail, time_tail))

        rms_excess = max(0.0, (rms - _OSC["rms_ok"]) / max(_OSC["rms_ok"], 1e-9))
        zc_excess  = max(0.0, (zc_rate - _OSC["zc_per_s_ok"]) / max(_OSC["zc_per_s_ok"], 1e-9))
        osc_pen = _OSC["k_rms"] * rms_excess + _OSC["k_zc"] * zc_excess
        mult *= (1.0 + osc_pen)
    else:
        rms = 0.0; zc_rate = 0.0

    # ----- NEW: steady-state bias & drift penalties (late segment)
    t_drift_start = max(t_end, T_end - _DRIFT["tail_s"])
    drift_mask = time >= t_drift_start
    if np.sum(drift_mask) >= 5:
        y = _moving_avg(v, int(max(1, 0.5/dt)))  # re-use smoothed baseline
        # linear trend
        p = np.polyfit(time[drift_mask], y[drift_mask], 1)
        slope = float(p[0])  # p.u. per second
        # late bias vs V_pre (prefer returning close to pre-drop)
        mean_late = float(np.mean(y[drift_mask]))
        bias = float(V_pre - mean_late)  # positive if controller drags down
        slope_excess = max(0.0, -(slope) - _DRIFT["slope_ok"]) / max(_DRIFT["slope_ok"], 1e-9)
        bias_excess  = max(0.0, (bias - _DRIFT["bias_ok"])) / max(_DRIFT["bias_ok"], 1e-9)
        drift_pen = _DRIFT["k_slope"] * slope_excess + _DRIFT["k_bias"] * bias_excess
        mult *= (1.0 + drift_pen)
    else:
        slope = 0.0; bias = 0.0

    fitness_value = mae_voltage * mult

    # debug summary
    print(f"Scenario: {scenario}")
    print(f"V_pre={V_pre:.4f}, drop t={drop_t:.3f}s, Vmin={min_v:.4f} @ {min_t:.3f}s")
    if steady_t is not None:
        print(f"Steady t={steady_t:.3f}s; hold≈{hold_len:.3f}s; in_envelope={in_envelope}")
    else:
        print(f"Steady not detected; hold≈{hold_len:.3f}s; in_envelope={in_envelope}")
    print(f"MAE={mae_voltage:.6g}  overshoot={overshoot_pu:.4g}  "
          f"rms_tail={rms:.4g}  zc/s={zc_rate:.3g}  slope={slope:.3g}  bias={bias:.4g}  "
          f"⇒ Fitness={fitness_value:.6g}")

    return float(fitness_value)

