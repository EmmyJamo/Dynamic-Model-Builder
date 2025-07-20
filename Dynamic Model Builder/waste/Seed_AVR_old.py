"""
avr_seed_from_snapshot.py

Generate 1st-cut AVR seed parameters for all generators flagged
`"selected_for_tuning": true` in the network snapshot JSON.

• Reads snapshot JSON (project-based path).
• For each selected generator:
      bus_for_analysis = Grid_Bus if Has_Trf else bus
      load <bus>.csv (header=0, skiprows=[1])
      extract dip+recovery features
      map to AVR seed (Ka, Ta, Tr, Ke, Te, Kf, Tf, Vrmax, Vrmin)
      write seed dict → generator["AVR_Seed"]
• Writes JSON back.

No writes into PowerFactory — this produces seed data for later ML tuning.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Config paths (edit if directory changes)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(r"C:\Users\james\OneDrive\MSc Project\results")
SNAPSHOT_BASE = (
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)


def _snapshot_path(pf_data) -> Path:
    return Path(SNAPSHOT_BASE) / f"{pf_data.project_name}_gen_snapshot.json"


# ---------------------------------------------------------------------------
# Seed container
# ---------------------------------------------------------------------------
@dataclass
class AVRSeed:
    Ka: float
    Ta: float
    Tr: float
    Ke: float
    Te: float
    Kf: float
    Tf: float
    Vrmax: float
    Vrmin: float
    note: str = "auto-seed v0"

    def as_dict(self) -> Dict[str, float]:
        return {
            "Ka": self.Ka,
            "Ta": self.Ta,
            "Tr": self.Tr,
            "Ke": self.Ke,
            "Te": self.Te,
            "Kf": self.Kf,
            "Tf": self.Tf,
            "Vrmax": self.Vrmax,
            "Vrmin": self.Vrmin,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------
def _load_snapshot(pf_data) -> dict:
    p = _snapshot_path(pf_data)
    if not p.exists():
        raise FileNotFoundError(f"Snapshot not found: {p}")
    with open(p, encoding="utf-8") as fp:
        return json.load(fp)


def _save_snapshot(pf_data, snap: dict) -> None:
    p = _snapshot_path(pf_data)
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(snap, fp, indent=2)
    print(f"💾 Snapshot updated: {p}")


# ---------------------------------------------------------------------------
# CSV read (your pattern: skip 2nd row)
# ---------------------------------------------------------------------------
def _load_response_csv(bus: str) -> pd.DataFrame:
    csv_path = RESULTS_DIR / f"{bus}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    print(f"📄 Loading CSV: {csv_path}")
    return pd.read_csv(csv_path, header=0, skiprows=[1])


def _select_columns(df: pd.DataFrame, bus: str) -> pd.DataFrame:
    # time
    tcol = None
    for c in ("All calculations",
              "All calculations (b:tnow in s)",
              "t", "time", "Time"):
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        raise RuntimeError("Time column not found.")

    # voltage
    if bus in df.columns:
        vcol = bus
    else:
        # fallback: find column with 'm:u1' marker
        vcol = next((c for c in df.columns if "m:u1" in c), None)
        if vcol is None:
            # fallback to any non-time numeric column
            vcol = next((c for c in df.columns if c != tcol), None)
    if vcol is None:
        raise RuntimeError("Voltage column not found.")

    out = pd.DataFrame({
        "t":   pd.to_numeric(df[tcol], errors="coerce"),
        "Vpu": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature extraction (dip + recovery)
# ---------------------------------------------------------------------------
def _extract_features(resp: pd.DataFrame,
                      dip_level: float = 0.9) -> Dict[str, Any]:
    t = resp["t"].to_numpy()
    v = resp["Vpu"].to_numpy()
    n = len(v)
    if n < 5:
        raise ValueError("Response trace too short.")

    # pre-disturbance ~ median last 10% of trace
    tail = max(1, n // 10)
    V_pre = float(np.median(v[-tail:]))

    # min voltage
    imin = int(np.argmin(v))
    V_min = float(v[imin])
    t_min = float(t[imin])

    # last time below dip threshold
    idx_below = np.where(v < dip_level)[0]
    t_release = float(t[idx_below[-1]]) if len(idx_below) else t_min

    # 63% recovery toward V_pre
    target_63 = dip_level + 0.63 * (V_pre - dip_level)
    idx_63 = np.where(v >= target_63)[0]
    t_63 = float(t[idx_63[0]]) if len(idx_63) else float("nan")

    # time to 0.95 & 0.98
    def _first_above(th):
        idx = np.where(v >= th)[0]
        return float(t[idx[0]]) if len(idx) else float("nan")

    t95 = _first_above(0.95)
    t98 = _first_above(0.98)

    # early slope (10–50 ms after release)
    w0 = t_release + 0.01
    w1 = t_release + 0.05
    seg = resp[(resp["t"] >= w0) & (resp["t"] <= w1)]
    slope = np.nan
    if len(seg) >= 2:
        slope = np.polyfit(seg["t"], seg["Vpu"], 1)[0]

    return {
        "V_pre": V_pre,
        "V_min": V_min,
        "t_min": t_min,
        "t_release": t_release,
        "t_63pct": t_63,
        "t_recov95": t95,
        "t_recov98": t98,
        "slope_early": slope,
    }


# ---------------------------------------------------------------------------
# Map features → seed values (heuristic)
# ---------------------------------------------------------------------------
def _seed_from_features(feat: Dict[str, Any]) -> AVRSeed:
    # Measurement delay ~ drop-to-release (bounded)
    Tr = feat["t_release"] - feat["t_min"]
    if not np.isfinite(Tr) or Tr <= 0:
        Tr = 0.02
    Tr = max(0.001, min(Tr, 0.1))

    # Controller time constant ~ recovery 63% window
    Ta = feat["t_63pct"] - feat["t_release"]
    if not np.isfinite(Ta) or Ta <= 0:
        Ta = 0.2
    Ta = max(0.02, min(Ta, 1.5))

    # Loop gain ~ early slope normalised (crude)
    slope = feat.get("slope_early", np.nan)
    if not np.isfinite(slope):
        Ka = 200.0
    else:
        Ka = slope * Ta / 0.1
        Ka = max(5.0, min(Ka, 500.0))

    # Generic placeholder values (refine later)
    Ke = 1.0
    Te = 0.5
    Kf = 0.0
    Tf = 1.0
    Vrmax = 5.0
    Vrmin = -5.0

    return AVRSeed(Ka=Ka, Ta=Ta, Tr=Tr,
                   Ke=Ke, Te=Te, Kf=Kf, Tf=Tf,
                   Vrmax=Vrmax, Vrmin=Vrmin)


# ---------------------------------------------------------------------------
# Public: build seeds for ALL selected generators (JSON only)
# ---------------------------------------------------------------------------
def build_seeds_from_snapshot(
        pf_data,
        dip_level,
        skip_non_sync: bool = True,
        require_csv: bool = True,
) -> Dict[str, AVRSeed]:
    """
    Generate AVR seeds for all snapshot generators flagged selected_for_tuning.

    Parameters
    ----------
    pf_data : PowerFactory wrapper
    dip_level : float
        Threshold used to detect end of dip (default 0.9 pu).
    skip_non_sync : bool
        If True, ignore non-synchronous gens; else include (same heuristic).
    require_csv : bool
        If True, skip gen when its CSV is missing; else create dummy seed.

    Returns
    -------
    Dict[str, AVRSeed]
        Mapping gen name → AVRSeed.
    """
    snap = _load_snapshot(pf_data)
    gens = snap.get("generators", [])
    todo = [g for g in gens if g.get("selected_for_tuning")]
    if not todo:
        print("No generators marked → nothing to seed.")
        return {}

    seeds: Dict[str, AVRSeed] = {}

    for meta in todo:
        gname = meta["name"]
        gtype = (meta.get("type") or "").lower()
        if skip_non_sync and gtype != "synchronous":
            print(f"⏭  Skip non-sync gen {gname}.")
            continue

        bus = meta.get("Grid_Bus") if meta.get("Has_Trf") else meta.get("bus")
        if not bus:
            print(f"⚠️  {gname}: no bus info → skip.")
            continue

        print(f"\n🔧 Seeding «{gname}»   (bus='{bus}')")

        # load CSV
        try:
            df_raw = _load_response_csv(bus)
        except FileNotFoundError as e:
            if require_csv:
                print(f"⚠️  CSV missing ({e}) → skip {gname}.")
                continue
            else:
                print(f"⚠️  CSV missing ({e}) → using dummy seed.")
                seed = AVRSeed(Ka=100, Ta=0.1, Tr=0.02,
                               Ke=1, Te=0.5, Kf=0, Tf=1,
                               Vrmax=5, Vrmin=-5, note="dummy (no CSV)")
                seeds[gname] = seed
                meta["AVR_Seed"] = seed.as_dict()
                continue

        # map & feature-extract
        try:
            resp = _select_columns(df_raw, bus)
            feat = _extract_features(resp, dip_level=dip_level)
        except Exception as e:
            print(f"⚠️  {gname}: feature extraction failed – {e}")
            continue

        # seed
        seed = _seed_from_features(feat)
        seeds[gname] = seed
        meta["AVR_Seed"] = seed.as_dict()
        print(f"   → Ka={seed.Ka:.1f} Ta={seed.Ta:.3f} Tr={seed.Tr:.3f}")

    # write back
    _save_snapshot(pf_data, snap)
    return seeds
