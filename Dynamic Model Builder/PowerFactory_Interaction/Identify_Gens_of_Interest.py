# ---------------------------------------------------------------------------
# pf_gen_impact.py   – ComVstab-based generator-impact scoring
# Python 3.9-compatible (→ no "|", no dataclasses, no f-string = specifiers)
# ---------------------------------------------------------------------------
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd


# ──────────────────────────────────────────
# 0)  constant paths / helpers
# ──────────────────────────────────────────
_SNAP_ROOT = (
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)


def _snapshot_path(pf_data):
    return Path(_SNAP_ROOT) / (pf_data.project_name + "_gen_snapshot.json")


def _get_bus_object(app, bus_name):
    """Return the ElmTerm whose loc_name == bus_name (or None)."""
    for b in app.GetCalcRelevantObjects("*.ElmTerm"):
        if b.GetAttribute("loc_name") == bus_name:
            return b
    return None


# ──────────────────────────────────────────
# 1)  single-bus ComVstab   → scalar score
# ──────────────────────────────────────────
def _run_comvstab_rms(app, bus_obj):
    """Run ComVstab at *bus_obj* and return RMS(|all sensitivities|)."""
    sen = app.GetFromStudyCase("ComVstab")
    if sen is None:
        raise RuntimeError("No ComVstab object in the study case.")
    sen.calcPtdf = 1
    sen.frmLimitsBrc = 0
    sen.p_bus = bus_obj
    if sen.Execute() != 0:
        raise RuntimeError("ComVstab failed at bus %s" % bus_obj.loc_name)

    res = app.GetCalcRelevantObjects("*.ElmRes")[0]
    res.Load()

    vals = []
    for c in range(res.GetNumberOfColumns()):
        ierr, v = res.GetValue(0, c)
        vals.append(0.0 if ierr else abs(v))
    res.Release()

    if not vals:
        return 0.0
    return float(np.sqrt(np.mean(np.square(vals))))


# ──────────────────────────────────────────
# 2)  main driver  (only AVR-equipped gens)
# ──────────────────────────────────────────
def run_generator_impact(pf_data, top_fraction=0.30, dry_run=False):

    # --- 2-a) read snapshot -------------------------------------------------
    snap_file = _snapshot_path(pf_data)
    with open(snap_file, encoding="utf-8") as fp:
        snap = json.load(fp)

    # -----------------------------------------------------------------------
    # Split the generator list once:
    #   • with_AVR  – candidates we’ll actually score
    #   • no_AVR    – ignored for impact ranking
    # -----------------------------------------------------------------------
    with_avr = [g for g in snap["generators"] if g.get("AVR")]
    no_avr   = [g for g in snap["generators"] if not g.get("AVR")]

    if not with_avr:
        raise RuntimeError("Snapshot has no generators with 'has_AVR': true")

    # -----------------------------------------------------------------------
    # Build a quick bus-name → ElmTerm map   (unchanged)
    # -----------------------------------------------------------------------
    app = pf_data.app
    all_buses = {b.GetAttribute("loc_name"): b
                 for b in app.GetCalcRelevantObjects("*.ElmTerm")}

    # -----------------------------------------------------------------------
    # 2-b) score ONLY the AVR-equipped machines
    # -----------------------------------------------------------------------
    scores = {}
    for g in with_avr:
        gname, bus_name = g["name"], g["bus"]
        bus_obj = all_buses.get(bus_name)
        if bus_obj is None:
            print(f"⚠️  Bus {bus_name} for {gname} not found; skipping")
            continue

        try:
            raw = _run_comvstab_rms(app, bus_obj)
        except RuntimeError as e:
            print("⚠️ ", e)
            continue

        # reactive head-room weighting  (as before)
        qmax, qmin = g.get("MVar_Max"), g.get("MVar_Min")
        qsch       = g.get("MVar_Sched", 0.0)
        head       = max(qmax - qsch, qsch - qmin) if None not in (qmax, qmin) else 1.0
        scores[gname] = raw * head

        print(f"{gname:<20} raw={raw:.3e}  head={head:.1f}  score={scores[gname]:.3e}")

    if not scores:
        raise RuntimeError("No scores computed – check model or ComVstab.")

    # -----------------------------------------------------------------------
    # 2-c) rank & flag -- top X % of AVR machines only
    # -----------------------------------------------------------------------
    ranked = sorted(scores, key=scores.get, reverse=True)
    top_n  = max(1, round(top_fraction * len(ranked)))
    keep   = set(ranked[:top_n])

    for g in snap["generators"]:
        nm = g["name"]
        g["VQ_score"] = scores.get(nm, 0.0)  # zero for no-AVR gens
        g["selected_for_tuning"] = nm in keep    # False for no-AVR gens

    # -----------------------------------------------------------------------
    # 2-d) meta info + write back
    # -----------------------------------------------------------------------
    snap["VQ_analysis_timestamp"] = datetime.now().isoformat(timespec="seconds")
    snap["VQ_method"]             = "ComVstab RMS × head-room (AVR gens only)"
    snap["VQ_top_fraction"]       = top_fraction
    snap["VQ_candidates"]         = len(with_avr)   # how many gens were eligible

    if dry_run:
        print("--dry-run--  JSON not updated.")
        return scores

    snap_file.write_text(json.dumps(snap, indent=2))
    print("✅  JSON updated →", snap_file)
