"""
bus_fitness_writer.py
─────────────────────
Iterate over every bus in the snapshot JSON, calculate the voltage‑control
fitness for *both* RMS disturbance scenarios, and write the results back
into the JSON.

Usage
-----
import bus_fitness_writer as BFW
BFW.update_bus_fitness(pf_data)
"""

from __future__ import annotations
from pathlib import Path
import json, traceback
import Data_Scoring.Voltage.V_P as EV


# ---------------------------------------------------------------------------
# Where snapshot JSONs live
# ---------------------------------------------------------------------------
_SNAP_DIR = (
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)

def _snap_path(pf_data) -> Path:
    return Path(_SNAP_DIR) / f"{pf_data.project_name}_gen_snapshot.json"


# ---------------------------------------------------------------------------
# Scenario → results‑folder mapping
# (⚠ adjust if you move the RMS output folders)
# ---------------------------------------------------------------------------
_SCEN_DIR = {
    "fast_dip":  r"C:\Users\james\OneDrive\MSc Project\results_2.2_rise",
    "slow_hold": r"C:\Users\james\OneDrive\MSc Project\results_7_rise",
}


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def update_bus_fitness(pf_data) -> None:
    """
    For every bus mentioned in the snapshot, evaluate
    ▸ fitness_fast_dip   (0.2‑s rise scenario)
    ▸ fitness_slow_hold  (7‑s rise scenario)
    and add/overwrite these keys in the JSON.

    If evaluation fails the corresponding value is set to null.
    """
    js_path = _snap_path(pf_data)
    if not js_path.exists():
        raise FileNotFoundError(js_path)

    with open(js_path, encoding="utf-8") as fp:
        snap = json.load(fp)

    # choose the section that lists buses
    if "buses" in snap:                       # preferred layout
        bus_records = snap["buses"]
        get_name    = lambda rec: rec["name"]
    elif "generators" in snap:                # legacy fallback
        bus_records = snap["generators"]
        get_name    = lambda rec: rec["bus"]
    else:
        raise KeyError("Neither 'buses' nor 'generators' found in snapshot")

    print(f"⚙️  Scoring {len(bus_records)} buses for two scenarios …")

    # ------------------------------------------------------------------ loop
    for rec in bus_records:
        bus = get_name(rec)

        # iterate over both scenarios
        for scen, folder in _SCEN_DIR.items():
            key = f"fitness_{scen}"
            try:
                score = EV.evaluate_voltage_control(bus, folder)
                rec[key] = float(score)           # force plain JSON number
                print(f"✓ {bus:20s}  {scen:10s} →  {score:.5g}")
            except Exception as err:
                print(f"⚠️  {bus} ({scen}): {err}")
                traceback.print_exc(limit=1)
                rec[key] = None

        # keep a legacy copy for the fast‑dip score if scripts expect it
        rec["original network fitness_value"] = rec["fitness_fast_dip"]

    # ------------------------------------------------------------------ write
    with open(js_path, "w", encoding="utf-8") as fp:
        json.dump(snap, fp, indent=2)

    print(f"\n✅  Fitness scores written to {js_path}")
