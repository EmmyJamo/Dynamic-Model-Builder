# ---------------------------------------------------------------------------
# gen_tuner.py  –  Generator-by‑generator AVR tuning loop
# ---------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
import json, time, datetime, shutil, traceback

# --- PowerFactory‑specific helpers ----------------------------------------
import PowerFactory_Interaction.Run_Initial_RMS     as R_RMS_Sim
import Data_Scoring.Voltage.V_P                     as EV
import PowerFactory_Interaction.Add_Seed_AVR_Values as Seed
import Machine_Learning.ML_DualHybrid_AVR_wave      as ML_wave
import Machine_Learning.ML_DualHybrid_AVR           as ML   # legacy single‑step

# ---------------------------------------------------------------------------
# static configuration
# ---------------------------------------------------------------------------
DUAL_MODE  = True        # use PSO + CMA dual wave
_MAX_ITER  = 12          # wrapper iterations per generator

_SCEN = {                # two RMS disturbance scenarios
    "fast_dip": {                                # 0.2‑s hold
        "rise_evt"     : "Voltage Rise 0.2s",
        "results_dir"  : r"C:\Users\james\OneDrive\MSc Project\results_2.2_rise",
        "switch_thresh": 8e-3,      # MAE (p.u.): promote to slow_hold when below
        "min_iters"    : 4,         # stay in fast phase at least this many loops
    },
    "slow_hold": {                              # 5‑s hold
        "rise_evt"     : "Voltage Rise 7s",
        "results_dir"  : r"C:\Users\james\OneDrive\MSc Project\results_7_rise",
    },
}

# single folder in which the scorer will always look
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
    """
    Copy the most recent ``<bus>.csv`` from the scenario‑specific results
    folder into ``_ML_RESULTS_DIR/<scenario>/<bus>.csv`` (over‑write OK).
    """
    src_dir = Path(_SCEN[scenario]["results_dir"]) / bus
    if not src_dir.exists():
        return

    latest = max(src_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, default=None)
    if not latest:
        return

    dest_dir = _ML_RESULTS_DIR / scenario
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, dest_dir / f"{bus}.csv")

# ---------------------------------------------------------------------------
# AVR capture helpers (unchanged)
# ---------------------------------------------------------------------------
_AVR_TAGS = ("Tr","Ka","Ta","Vrmax","Vrmin","Ke","Te","Kf","Tf","E1","E2","Se1","Se2")
def _find_avr_block(pf_data, meta:dict):
    avr_hint = meta.get("AVR_Name")
    try:
        blks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
    except Exception:
        return None
    if avr_hint:
        for b in blks:
            if b.loc_name == avr_hint:
                return b
    g_l = meta["name"].lower()
    for b in blks:
        ln = b.loc_name.lower()
        if "avr" in ln and g_l in ln:
            return b
    for b in blks:
        if "avr" in b.loc_name.lower():
            return b
    return None

def _capture_avr_params(pf_data, meta):
    blk, out = _find_avr_block(pf_data, meta), {}
    if not blk: return out
    for t in _AVR_TAGS:
        for tag in (f"e:{t}", t):
            try:
                v = blk.GetAttribute(tag)
                if v is not None:
                    out[t] = float(v); break
            except Exception: pass
    return out

# ---------------------------------------------------------------------------
# basic PF helpers
# ---------------------------------------------------------------------------
def _activate_variant(pf_data, gname:str):
    scheme = pf_data.variations_folder.GetContents(f"TUNE_{gname}.IntScheme")[0]
    scheme.Activate()
    print(f"   ✓ variant «{scheme.loc_name}» activated")

def _enable_param_events(pf_data, bus:str, scenario:str):
    want = _SCEN[scenario]["rise_evt"]
    other = _SCEN["slow_hold" if scenario=="fast_dip" else "fast_dip"]["rise_evt"]
    for name, out in ((f"Voltage Drop {bus}.EvtParam",            False),
                      (f"{want} {bus}.EvtParam",                  False),
                      (f"{other} {bus}.EvtParam",                 True )):
        lst = pf_data.Simulation_Folder.GetContents(name)
        if lst: lst[0].SetAttribute("e:outserv", out)

def _set_voltage_source(pf_data, bus, R, X, U0):
    vac = pf_data.grid_folder.GetContents(f"{bus}V_Source.ElmVac")[0]
    vac.e_R1, vac.e_X1, vac.e_vtarget = R, X, U0

# ---------------------------------------------------------------------------
# candidate evaluation (PSO / CMA)
# ---------------------------------------------------------------------------
def _eval_candidate_vec(pf_data, gname, bus, vec, label, scenario):
    ML_wave.write_candidate(pf_data, gname, vec)
    R_RMS_Sim.quick_rms_run(
        pf_data, "Power Flow", bus, _SCEN[scenario]["results_dir"], None
    )
    _promote_latest_csv(bus, scenario)
    time.sleep(0.25)                      # allow PF to flush file
    score, *_ = EV.evaluate_voltage_control(bus)
    print(f"      ↩ {label} score = {score:.6g}")
    return score

# ---------------------------------------------------------------------------
# MAIN entry
# ---------------------------------------------------------------------------
def tune_selected_generators(pf_data,
                             target_score: float = 8.5e-4,
                             max_iter: int = _MAX_ITER,
                             dry_run: bool = False):

    snap_p = _snap_path(pf_data); snap = json.loads(snap_p.read_text())
    todo   = [g for g in snap["generators"] if g.get("selected_for_tuning")]

    if not todo:
        print("📭  No generators marked for tuning."); return

    for meta in todo:
        gname = meta["name"]
        bus   = meta["Grid_Bus"] if meta.get("Has_Trf") else meta["bus"]
        R,X,_,U0 = R_RMS_Sim.get_bus_thevenin(pf_data, bus)

        # --- setup ---------------------------------------------------------
        _activate_variant(pf_data, gname)
        _set_voltage_source(pf_data, bus, R, X, U0)
        Seed._seed_avr_parameters(pf_data, meta, gname)
        ML_wave.prepare_hybrid(pf_data, meta)              # fresh optimiser

        scenario  = "fast_dip"
        best_score, fast_hist = float("inf"), []

        for k in range(1, max_iter+1):
            print(f"\n[{gname}] iteration {k}/{max_iter}  –  scenario = {scenario}")

            _enable_param_events(pf_data, bus, scenario)

            # ----- PSO & CMA candidates -----------------------------------
            cand_pso, cand_cma = ML_wave.ask_two(gname)

            sc_p = _eval_candidate_vec(pf_data, gname, bus, cand_pso, "PSO",  scenario)
            sc_c = _eval_candidate_vec(pf_data, gname, bus, cand_cma, "CMA",  scenario)

            best_src, best_vec = ML_wave.tell_dual(
                gname, sc_p, sc_c, cand_pso=cand_pso, cand_cma=cand_cma,
                scenario=scenario                       # <‑‑ extra flag
            )
            ML_wave.write_candidate(pf_data, gname, best_vec)
            best_score = min(sc_p, sc_c)

            # ----- decision: promote to slow_hold? -------------------------
            if scenario == "fast_dip":
                fast_hist.append(best_score)
                if (k >= _SCEN["fast_dip"]["min_iters"]
                        and sum(fast_hist[-3:]) / len(fast_hist[-3:])
                             <= _SCEN["fast_dip"]["switch_thresh"]):
                    print("   🔁  criteria met – switching to slow_hold.")
                    scenario = "slow_hold"
                    ML_wave.prepare_hybrid(pf_data, meta)   # reset optimiser
                    continue

            # ----- convergence? -------------------------------------------
            if best_score <= target_score:
                print(f"   🎯  target achieved (score {best_score:.6g})")
                break

        # ----- capture & persist results ----------------------------------
        meta["AVR_Final"]     = _capture_avr_params(pf_data, meta)
        meta["final_score"]   = best_score
        meta["tuned_iter"]    = k
        meta["tuning_ts"]     = datetime.datetime.now().isoformat(timespec="seconds")

    snap_p.write_text(json.dumps(snap, indent=2))
    print("\n✅  Tuning finished – snapshot JSON updated")
