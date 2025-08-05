# ---------------------------------------------------------------------------
# gen_tuner.py  –  Generator‑by‑generator AVR tuning loop (PSO only)
# ---------------------------------------------------------------------------
from __future__ import annotations
import json, time, datetime, shutil, traceback
from pathlib import Path

# ---------- PowerFactory helpers ------------------------------------------
import PowerFactory_Control.Run_RMS_Sim             as R_RMS_Sim
import Data_Scoring.Voltage.V_P                     as EV
import PowerFactory_Interaction.Add_Seed_AVR_Values as Seed
import PowerFactory_Interaction.Run_Initial_RMS     as Run_In_RMS

# ---------- NEW: lightweight PSO engine -----------------------------------
from Machine_Learning import ML_PSO_AVR as PSO

# ---------------------------------------------------------------------------
# static configuration
# ---------------------------------------------------------------------------
_MAX_ITER = 12               # wrapper iterations per generator

_SCEN = {                    # two RMS voltage‑disturbance scenarios
    "fast_dip": {
        "rise_evt"     : "Voltage Rise 0.2s",
        "results_dir"  : r"C:\Users\james\OneDrive\MSc Project\results_2.2_rise",
        "switch_thresh": 8e-3,      # MAE  → promote to slow_hold when below
        "min_iters"    : 4,
    },
    "slow_hold": {
        "rise_evt"     : "Voltage Rise 7s",
        "results_dir"  : r"C:\Users\james\OneDrive\MSc Project\results_7_rise",
    },
}

_ML_RESULTS_DIR = Path(r"C:\Users\james\OneDrive\MSc Project\results_ml")

# ---------------------------------------------------------------------------
# snapshot helpers
# ---------------------------------------------------------------------------
_SNAP_BASE = (r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
              r"\Dynamic Model Builder\JSON_DB\Network_Snapshots")
def _snap_path(pf_data) -> Path:
    return Path(_SNAP_BASE) / f"{pf_data.project_name}_gen_snapshot.json"

# ---------------------------------------------------------------------------
# quick CSV promotion so the scorer can work in a fixed directory
# ---------------------------------------------------------------------------
def _promote_latest_csv(bus: str, scenario: str) -> None:
    src_dir = Path(_SCEN[scenario]["results_dir"]) / bus
    if not src_dir.exists():
        return
    latest = max(src_dir.glob("*.csv"),
                 key=lambda p: p.stat().st_mtime,
                 default=None)
    if latest:
        dest = _ML_RESULTS_DIR / scenario
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest, dest / f"{bus}.csv")

# ---------------------------------------------------------------------------
# basic PowerFactory helpers
# ---------------------------------------------------------------------------
def _activate_variant(pf_data, gname: str) -> None:
    scheme = pf_data.variations_folder.GetContents(f"TUNE_{gname}.IntScheme")[0]
    scheme.Activate()
    print(f"   ✓ variant «{scheme.loc_name}» activated")

def _deactivate_variant(pf_data, gname: str) -> None:
    scheme = pf_data.variations_folder.GetContents(f"TUNE_{gname}.IntScheme")[0]
    scheme.Deactivate()
    print(f"   ✓ variant «{scheme.loc_name}» activated")

def _enable_param_events(pf_data, bus: str, scenario: str) -> None:
    want  = _SCEN[scenario]["rise_evt"]
    other = _SCEN["slow_hold" if scenario == "fast_dip" else "fast_dip"]["rise_evt"]
    for obj, out in ((f"Voltage Drop {bus}.EvtParam", False),
                     (f"{want} {bus}.EvtParam",       False),
                     (f"{other} {bus}.EvtParam",      True )):
        lst = pf_data.Simulation_Folder.GetContents(obj)
        if lst:
            lst[0].SetAttribute("e:outserv", out)

def _set_voltage_source(pf_data, bus: str, R, X, U0) -> None:
    vac = pf_data.grid_folder.GetContents(f"{bus}V_Source.ElmVac")[0]
    vac.SetAttribute("e:usetp", U0)
    vac.SetAttribute("e:R1", R)
    vac.SetAttribute("e:X1", X)
    vac.SetAttribute("e:outserv", False)  # nominal voltage

# ---------------------------------------------------------------------------
# AVR read‑back (unchanged)
# ---------------------------------------------------------------------------
_AVR_TAGS = ("Tr","Ka","Ta","Vrmax","Vrmin","Ke","Te",
             "Kf","Tf","E1","E2","Se1","Se2")
def _find_avr_block(pf_data, meta):
    hint = (meta.get("AVR_Name") or "").lower()
    blks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
    if hint:
        for b in blks:
            if b.loc_name.lower() == hint: return b
    g_l = meta["name"].lower()
    for b in blks:
        ln = b.loc_name.lower()
        if "avr" in ln and g_l in ln: return b
    return next((b for b in blks if "avr" in b.loc_name.lower()), None)

def _capture_avr_params(pf_data, meta):
    blk = _find_avr_block(pf_data, meta); out = {}
    if not blk: return out
    for t in _AVR_TAGS:
        for tag in (f"par_{t}", f"e:{t}", t):
            try:
                v = blk.GetAttribute(tag)
                if v is not None:
                    out[t] = float(v); break
            except Exception:
                pass
    return out

# --------------------------------------------------------------------------
# evaluate ONE PSO candidate and return its fitness
# --------------------------------------------------------------------------
def _eval_candidate(pf_data,
                    gname:    str,
                    bus:      str,
                    vec:      list[float],
                    scenario: str) -> float:
    """
    • writes *vec* into the AVR,
    • runs the RMS disturbance for the requested scenario,
    • copies the latest CSV into the fixed ML results folder,
    • calls the voltage‑fitness scorer with the correct folder path,
    • returns the scalar fitness (lower = better).
    """
    # 1) push parameters into PF
    PSO.write_candidate(pf_data, gname, vec)

    # 2) run RMS using the scenario‑specific results directory
    R_RMS_Sim.quick_rms_run(
        pf_data,
        "Power Flow",
        bus,
        _SCEN[scenario]["results_dir"],
        None
    )

    # 3) copy the freshly‑written CSV to a stable location for the scorer
    _promote_latest_csv(bus, scenario)

    # 4) give PF a moment to finish file I/O
    time.sleep(0.25)

    # 5) score using the folder the scorer expects
    csv_root = _ML_RESULTS_DIR / scenario          # <‑‑ NEW
    score = EV.evaluate_voltage_control(bus, csv_root)

    print(f"      ↩ fitness = {score:.6g}")
    return score


# ---------------------------------------------------------------------------
# MAIN entry
# ---------------------------------------------------------------------------
def tune_selected_generators(pf_data,
                             target_score: float = 8.5e-4,
                             max_iter: int   = _MAX_ITER,
                             dry_run:   bool = False):

    snap_p = _snap_path(pf_data)
    snap   = json.loads(snap_p.read_text())
    todo   = [g for g in snap["generators"] if g.get("selected_for_tuning")]

    if not todo:
        print("📭  No generators marked for tuning."); return

    for meta in todo:
        gname = meta["name"]
        bus   = meta["Grid_Bus"] if meta.get("Has_Trf") else meta["bus"]
        R, X, _, U0 = Run_In_RMS.get_bus_thevenin(pf_data, bus)

        # ---------- setup variant & initial state -------------------------
        _activate_variant(pf_data, gname)
        _set_voltage_source(pf_data, bus, R, X, U0)
        Seed._seed_avr_parameters(pf_data, meta, gname)
        PSO.prepare_pso(pf_data, meta, n_particles=10)

        scenario  = "fast_dip"
        fast_hist: list[float] = []
        best_score = float("inf")

        for k in range(1, max_iter + 1):
            print(f"\n[{gname}] iteration {k}/{max_iter}  •  scenario = {scenario}")
            _enable_param_events(pf_data, bus, scenario)

            # -------- ask, evaluate, tell --------------------------------
            cand   = PSO.ask_one(gname)
            score  = _eval_candidate(pf_data, gname, bus, cand, scenario)
            PSO.tell_one(gname, score, cand)
            best_score = min(best_score, score)

            # -------- scenario switch logic ------------------------------
            if scenario == "fast_dip":
                fast_hist.append(score)
                if (k >= _SCEN["fast_dip"]["min_iters"]
                        and sum(fast_hist[-3:]) / len(fast_hist[-3:])
                                <= _SCEN["fast_dip"]["switch_thresh"]):
                    print("   🔁 switching to slow_hold scenario")
                    scenario = "slow_hold"
                    PSO.prepare_pso(pf_data, meta)   # fresh swarm
                    continue

            # -------- convergence? ---------------------------------------
            if best_score <= target_score:
                print(f"   🎯 target achieved (score {best_score:.6g})")
                _deactivate_variant(pf_data, gname)
                break

        # ---------- save best vector back to PF & snapshot ---------------
        PSO.write_candidate(pf_data, gname, PSO.get_best(gname))
        meta["AVR_Final"]   = _capture_avr_params(pf_data, meta)
        meta["final_score"] = best_score
        meta["tuned_iter"]  = k
        meta["tuning_ts"]   = datetime.datetime.now().isoformat(timespec="seconds")

    snap_p.write_text(json.dumps(snap, indent=2))
    print("\n✅  Tuning finished – snapshot JSON updated")

