"""
avr_seed_from_snapshot.py ‑‑ dual‑scenario version
--------------------------------------------------
Derives first‑cut AVR seed values **only from bus‑voltage traces** recorded
in two RMS simulations:
  • fast_dip   : 10 % drop, release after 0.2 s
  • slow_hold  : 10 % drop, hold for 5 s, then rise
No AVR internals required.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, numpy as np, pandas as pd
from typing import Dict, Any, Tuple, Union

# ───────── paths ───────────────────────────────────────────────────────────
RESULTS_DIRS = {
    "fast_dip":  Path(r"C:\Users\james\OneDrive\MSc Project\results_2.2_rise"),
    "slow_hold": Path(r"C:\Users\james\OneDrive\MSc Project\results_7_rise"),
}
SNAPSHOT_BASE = (
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)


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

# ───────── helper to locate snapshot json ──────────────────────────────────
def _snapshot_path(pf_data) -> Path:
    return Path(SNAPSHOT_BASE) / f"{pf_data.project_name}_gen_snapshot.json"

# ───────── seed container ──────────────────────────────────────────────────
@dataclass
class AVRSeed:
    Ka: float; Ta: float; Tr: float
    Ke: float; Te: float; Kf: float; Tf: float
    Vrmax: float; Vrmin: float
    note: str = "auto‑seed v1 (dual‑scenario)"
    def as_dict(self) -> Dict[str, Union[str, float]]:
        d = self.__dict__.copy(); return d

# ───────── generic CSV loader (per scenario) ───────────────────────────────
def _load_csv(bus: str, scenario: str) -> pd.DataFrame:
    base = RESULTS_DIRS[scenario]
    # exporter path pattern: <folder>/<Bus XX>/<Bus XX>.csv
    p = base / bus / f"{bus}.csv"
    if not p.exists():                    # fallback flat pattern
        p = base / f"{bus}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, header=0, skiprows=[1])

# ───────── time & voltage column picker ────────────────────────────────────
def _select_tv(df: pd.DataFrame, bus: str) -> pd.DataFrame:
    tcol = next((c for c in df.columns
                 if c.lower().startswith(("all cal","t","time"))), None)
    if tcol is None: raise RuntimeError("time col not found")
    vcol = bus if bus in df.columns else \
           next((c for c in df.columns if "m:u1" in c), None)
    if vcol is None: vcol = next(c for c in df.columns if c != tcol)
    return (pd.DataFrame({"t":pd.to_numeric(df[tcol], errors="coerce"),
                          "Vpu":pd.to_numeric(df[vcol], errors="coerce")})
            .dropna().reset_index(drop=True))

# ───────── feature extractors ──────────────────────────────────────────────
def _feat_fast(resp: pd.DataFrame, dip=0.9) -> Dict[str,float]:
    t,v = resp["t"].to_numpy(), resp["Vpu"].to_numpy()
    Vpre = float(np.median(v[-max(1,len(v)//10):]))
    i_min = int(np.argmin(v)); t_min, Vmin = float(t[i_min]), float(v[i_min])
    idx_below = np.where(v<dip)[0]
    t_rel = float(t[idx_below[-1]]) if len(idx_below) else t_min
    target63 = dip + 0.63*(Vpre-dip)
    idx63 = np.where(v>=target63)[0]
    t63 = float(t[idx63[0]]) if len(idx63) else np.nan
    seg = resp[(resp.t>=t_rel+0.01)&(resp.t<=t_rel+0.05)]
    slope = np.polyfit(seg.t, seg.Vpu,1)[0] if len(seg)>=2 else np.nan
    return dict(Vpre=Vpre,Vmin=Vmin,t_min=t_min,t_rel=t_rel,t63=t63,slope=slope)

def _feat_hold(resp: pd.DataFrame, dip=0.9,
               t_drop=2.0,t_rise=7.0) -> Dict[str,float]:
    t,v=resp.t.to_numpy(),resp.Vpu.to_numpy()
    Vpre=float(np.median(v[t < t_drop-0.05]))
    win_end=(t>=t_rise-0.5)&(t<=t_rise-0.05)
    Vend=float(np.median(v[win_end])) if win_end.any() else np.nan
    noise=float(np.std(v[win_end])) if win_end.any() else 0.0
    return dict(Vpre=Vpre,Vend=Vend,noise=noise,dip=dip)

# ───────── fusion heuristic (fast+hold) ────────────────────────────────────
def _seed_from_dual(f:dict,h:dict|None,avr:dict) -> AVRSeed:
    # Tr, Ta, Ka from fast
    Tr=max(0.001,min(max(f["t_rel"]-f["t_min"],0.02),0.1))
    Ta=max(0.02,min(max(f["t63"]-f["t_rel"],0.2),1.5))
    slope=f["slope"]; Ka = 200 if np.isnan(slope) else np.clip(slope*Ta/0.1,5,500)

    # long‑hold scaling
    if h and np.isfinite(h["Vend"]):
        RF = (h["Vend"]-h["dip"])/(h["Vpre"]-h["dip"]+1e-6)
        G = np.clip(1/max(RF,1e-3),0.5,5)
        Ka = np.clip(0.6*Ka + 0.4*Ka*G,5,500)
        Te = 0.5 if RF>0.9 else np.clip(0.5+2*(1-RF),0.1,1.5)
        Kf = 0.08 if h["noise"]>0.002 else np.clip(0.02*Ka/100,0,0.1)
    else:
        Te = 0.5; Kf = np.clip(0.02*Ka/100,0,0.1)

    # remaining params from PF or defaults
    Vrmax = avr.get("Vrmax",5.0); Vrmin = avr.get("Vrmin",-5.0)
    Ke = avr.get("Ke", 1/max(abs(Vrmax),0.2)); Tf = 1.0

    return AVRSeed(Ka,Ta,Tr,Ke,Te,Kf,Tf,Vrmax,Vrmin)

# ───────── public entry ‑ main loop ────────────────────────────────────────
def build_seeds_from_snapshot(pf_data,dip_level=0.9,
                              t_drop=2.0,t_rise_short=2.2,t_rise_long=7.0,
                              require_csv=True) -> Dict[str,AVRSeed]:
    snap=_load_snapshot(pf_data); seeds={}
    for g in snap.get("generators",[]):
        if not g.get("selected_for_tuning"): continue
        bus=g.get("Grid_Bus") if g.get("Has_Trf") else g.get("bus")
        if not bus: continue
        print(f"\n🔧 {g['name']}  (bus {bus})")

        try:
            df_fast=_select_tv(_load_csv(bus,"fast_dip"),bus)
            f=_feat_fast(df_fast,dip_level)
        except Exception as e:
            print(f"⚠ fast‑dip missing/failed → {e}")
            if require_csv: continue
            f=dict(t_rel=0,t_min=0,t63=0,slope=np.nan)

        h=None
        try:
            df_hold=_select_tv(_load_csv(bus,"slow_hold"),bus)
            h=_feat_hold(df_hold,dip_level,t_drop,t_rise_long)
        except Exception as e:
            print(f"   (no slow‑hold) {e}")

        seed=_seed_from_dual(f,h,g.get("AVR_Params",{}))
        seeds[g["name"]]=seed; g["AVR_Seed"]=seed.as_dict()
        print(f"   → Ka={seed.Ka:.1f}  Ta={seed.Ta:.3f}  Tr={seed.Tr:.3f} "
              f"Te={seed.Te:.3f}  Kf={seed.Kf:.3f}")

    _save_snapshot(pf_data,snap)
