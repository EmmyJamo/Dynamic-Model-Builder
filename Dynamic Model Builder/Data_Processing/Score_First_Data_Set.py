"""
bus_fitness_writer.py
─────────────────────
Iterate over every bus in the snapshot JSON, call
EV.evaluate_voltage_control(bus_name) and add the resulting score
to the JSON in-place.

Usage
-----
import bus_fitness_writer as BFW
BFW.update_bus_fitness(pf_data)
"""

from __future__ import annotations
from pathlib import Path
import json
import traceback
import Data_Scoring.Voltage.V_P as EV        


# ------------------------------------------------------------------
_SNAP_DIR = (
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)


def _snap_path(pf_data) -> Path:
    """Return Path to the generator snapshot JSON for *pf_data* project"""
    return Path(_SNAP_DIR) / f"{pf_data.project_name}_gen_snapshot.json"


# ------------------------------------------------------------------
# main entry point
# ------------------------------------------------------------------
def update_bus_fitness(pf_data) -> None:
    """
    Add / overwrite `"fitness_value"` for every bus in the snapshot file.
    """
    js_path = _snap_path(pf_data)
    if not js_path.exists():
        raise FileNotFoundError(js_path)

    with open(js_path, encoding="utf-8") as fp:
        snap = json.load(fp)

    # pick the list of buses
    if "buses" in snap:                       # preferred layout
        bus_records = snap["buses"]
        get_name = lambda rec: rec["name"]
    elif "generators" in snap:                # legacy fallback
        bus_records = snap["generators"]
        get_name = lambda rec: rec["bus"]
    else:
        raise KeyError("Neither 'buses' nor 'generators' sections found")

    print(f"⚙️  Scoring {len(bus_records)} buses …")

    # ------------------------------------------------------------------
    for rec in bus_records:
        bus = get_name(rec)
        try:
            fitness = EV.evaluate_voltage_control(bus)
            rec["fitness_value"] = float(fitness)  # ensure plain JSON number
            print(f"✓ {bus:20s}  →  score = {fitness:.5g}")
        except Exception as err:
            # keep going, but record the failure
            print(f"⚠️  {bus}: {err}")
            traceback.print_exc(limit=1)
            rec["fitness_value"] = None

    # ------------------------------------------------------------------
    # save back – keep other keys (generators, meta-timestamps, …) intact
    with open(js_path, "w", encoding="utf-8") as fp:
        json.dump(snap, fp, indent=2)

    print(f"\n✅  Fitness scores written to {js_path}")
