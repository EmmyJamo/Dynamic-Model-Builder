# ---------------------------------------------------------------------------
# gen_tuner.py  –  Generator-by-generator AVR tuning loop (PSO only)
# ---------------------------------------------------------------------------
from __future__ import annotations
import json, time, datetime
from pathlib import Path

# plotting (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- PowerFactory helpers ------------------------------------------
import PowerFactory_Control.Run_RMS_Sim             as R_RMS_Sim
import Data_Scoring.Voltage.V_P                     as EV
import PowerFactory_Interaction.Add_Seed_AVR_Values as Seed
import PowerFactory_Interaction.Run_Initial_RMS     as Run_In_RMS

# ---------- PSO engine -----------------------------------------------------
from Machine_Learning import ML_PSO_AVR as PSO

# ---------------------------------------------------------------------------
# static configuration
# ---------------------------------------------------------------------------
_MAX_ITER = 12

_SCEN = {
    "fast_dip": {
        "rise_evt"     : "Voltage Rise 0.2s",
        "results_dir"  : str(Path(r"C:\Users\james\OneDrive\MSc Project\results_ml") / "fast_dip"),
        "switch_thresh": 8e-3,
        "min_iters"    : 4,
    },
    "slow_hold": {
        "rise_evt"     : "Voltage Rise 7s",
        "results_dir"  : str(Path(r"C:\Users\james\OneDrive\MSc Project\results_ml") / "slow_hold"),
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
# basic PowerFactory helpers
# ---------------------------------------------------------------------------
def _activate_variant(pf_data, gname: str) -> str:
    scheme = pf_data.variations_folder.GetContents(f"TUNE_{gname}.IntScheme")[0]
    scheme.Activate()
    print(f"   ✓ variant «{scheme.loc_name}» activated")
    return scheme.loc_name

def _deactivate_variant(pf_data, var_name: str) -> None:
    scheme = pf_data.variations_folder.GetContents(f"{var_name}.IntScheme")[0]
    scheme.Deactivate()
    print(f"   ✓ variant «{scheme.loc_name}» deactivated")

def _enable_param_events(pf_data, bus: str, scenario: str) -> None:
    want  = _SCEN[scenario]["rise_evt"]
    other = _SCEN["slow_hold" if scenario == "fast_dip" else "fast_dip"]["rise_evt"]
    for obj, out in ((f"Voltage Drop {bus}.EvtParam", False),
                     (f"{want} {bus}.EvtParam",       False),
                     (f"{other} {bus}.EvtParam",      True )):
        lst = pf_data.Simulation_Folder.GetContents(obj)
        if lst: lst[0].SetAttribute("e:outserv", out)

def _set_voltage_source(pf_data, bus: str, R, X, U0) -> None:
    vac = pf_data.grid_folder.GetContents(f"{bus}V_Source.ElmVac")[0]
    vac.SetAttribute("e:usetp", U0)
    vac.SetAttribute("e:R1", R)
    vac.SetAttribute("e:X1", X)
    vac.SetAttribute("e:outserv", False)

# ---------------------------------------------------------------------------
# AVR read-back (unchanged)
# ---------------------------------------------------------------------------
_AVR_TAGS = ("Tr","Ka","Ta","Vrmax","Vrmin","Ke","Te","Kf","Tf","E1","E2","Se1","Se2")
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
            except Exception: pass
    return out

# ---------------------------------------------------------------------------
# fitness history (per variation)
# ---------------------------------------------------------------------------
def _save_plot_for_variation(var_name: str, scores: list[float],
                             out_root: Path = _ML_RESULTS_DIR / "plots") -> Path | None:
    if not scores:
        return None
    out_dir = out_root / var_name.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    x = list(range(1, len(scores)+1))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, scores, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness (lower is better)")
    ax.set_title(f"{var_name} – fitness per iteration")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / "fitness.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"   📈 saved plot → {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# evaluate ONE PSO candidate and return its fitness
# ---------------------------------------------------------------------------
def _eval_candidate(pf_data,
                    meta:     dict,
                    bus:      str,
                    vec:      list[float],
                    scenario: str) -> float:
    # 1) write the candidate to PF (uses bound AVR handle)
    if not PSO.write_candidate(pf_data, meta, vec):
        print("      ⚠️ write_candidate returned False")

    # 2) run RMS – results go directly into …/results_ml/<scenario>/
    out_dir = Path(_SCEN[scenario]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    R_RMS_Sim.quick_rms_run(pf_data, "Power Flow", bus, str(out_dir), None)

    # 3) brief pause to let PF close the file
    time.sleep(0.25)

    # 4) score – the scorer reads the same directory
    score = EV.evaluate_voltage_control(bus, out_dir)
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
        var_name = _activate_variant(pf_data, gname)  # capture variation name
        _set_voltage_source(pf_data, bus, R, X, U0)

        # seed → bind the exact AVR → prepare PSO
        Seed._seed_avr_parameters(pf_data, meta, gname)
        if not PSO.bind_avr(pf_data, meta):
            print(f"   ⚠️ could not bind AVR for «{gname}». Skipping.")
            _deactivate_variant(pf_data, var_name)
            continue
        PSO.prepare_pso(pf_data, meta, n_particles=10)

        scenario   = "fast_dip"
        var_scores: list[float] = []
        best_score = float("inf")

        for k in range(1, max_iter + 1):
            print(f"\n[{gname}/{var_name}] iteration {k}/{max_iter} • scenario = {scenario}")
            _enable_param_events(pf_data, bus, scenario)

            # -------- ask, evaluate, tell --------------------------------
            cand   = PSO.ask_one(gname)
            score  = _eval_candidate(pf_data, meta, bus, cand, scenario)
            PSO.tell_one(gname, score, cand)
            best_score = min(best_score, score)

            # record score for THIS VARIATION
            var_scores.append(float(score))

            # -------- convergence? ---------------------------------------
            if best_score <= target_score:
                print(f"   🎯 target achieved (score {best_score:.6g})")
                _deactivate_variant(pf_data, var_name)
                break

            if k == _MAX_ITER:
                print(f"   ⏳ max iterations reached (score {best_score:.6g})")
                _deactivate_variant(pf_data, var_name)
                break

        # ---------- save best vector back to PF & snapshot ---------------
        PSO.write_candidate(pf_data, meta, PSO.get_best(gname))
        meta["AVR_Final"]   = _capture_avr_params(pf_data, meta)
        meta["final_score"] = best_score
        meta["tuned_iter"]  = k
        meta["tuning_ts"]   = datetime.datetime.now().isoformat(timespec="seconds")

        # ---------- save fitness plot FOR THIS VARIATION -----------------
        _save_plot_for_variation(var_name, var_scores)

    snap_p.write_text(json.dumps(snap, indent=2))
    print("\n✅  Tuning finished – snapshot JSON updated")
