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
from Machine_Learning.ML_PSO_AVR import TUNED_TAGS  # to label best-vector CSV

# ---------------------------------------------------------------------------
# static configuration
# ---------------------------------------------------------------------------
_MAX_ITER = 60  # allow more iterations; plateau logic will stop earlier

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

# Convergence (plateau) detection on the score stream
_STOP_CRIT = {
    "min_iters": 10,      # never stop before this many iterations
    "window":     6,      # use the last W scores against the previous W
    "eps_abs":  2e-4,     # absolute improvement threshold
    "eps_rel":  0.01,     # OR 1% relative improvement threshold
}

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
# fitness history helpers (per variation): CSV + PNG updated each iteration
# ---------------------------------------------------------------------------
def _var_plot_dir(var_name: str) -> Path:
    return (_ML_RESULTS_DIR / "plots" / var_name.replace(" ", "_"))

def _write_history_csv(var_name: str, scores: list[float]) -> Path:
    d = _var_plot_dir(var_name); d.mkdir(parents=True, exist_ok=True)
    p = d / "fitness_history.csv"
    with p.open("w", encoding="utf-8") as fp:
        fp.write("iteration,fitness\n")
        for i, s in enumerate(scores, 1):
            fp.write(f"{i},{float(s)}\n")
    return p

def _save_plot_for_variation(var_name: str, scores: list[float]) -> Path | None:
    if not scores: return None
    out_dir = _var_plot_dir(var_name)
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
    return out_path

def _update_progress_artifacts(var_name: str, scores: list[float]) -> None:
    csv_p  = _write_history_csv(var_name, scores)
    png_p  = _save_plot_for_variation(var_name, scores)
    print(f"   📊 progress: {len(scores)} pts → {csv_p}")
    if png_p: print(f"   📈 plot updated → {png_p}")

# ---- best-vector persistence ----------------------------------------------
def _write_best_vector_csv(var_name: str, best_score: float, best_iter: int, vec: list[float]) -> Path:
    d = _var_plot_dir(var_name); d.mkdir(parents=True, exist_ok=True)
    p = d / "best_vector.csv"
    with p.open("w", encoding="utf-8") as fp:
        fp.write("best_iter,best_score," + ",".join(TUNED_TAGS) + "\n")
        vals = ",".join(str(float(v)) for v in vec)
        fp.write(f"{best_iter},{best_score},{vals}\n")
    print(f"   📝 best-vector saved → {p}")
    return p

# ---------------------------------------------------------------------------
# plateau detection on score stream
# ---------------------------------------------------------------------------
def _mean(vals: list[float]) -> float:
    return sum(vals) / max(1, len(vals))

def _has_plateau(scores: list[float]) -> bool:
    """
    True  -> stop: recent improvement is negligible
    False -> keep going
    """
    w   = _STOP_CRIT["window"]
    mi  = _STOP_CRIT["min_iters"]
    ea  = _STOP_CRIT["eps_abs"]
    er  = _STOP_CRIT["eps_rel"]

    if len(scores) < max(mi, 2*w):
        return False

    recent = scores[-w:]
    prev   = scores[-2*w:-w]
    prev_m = _mean(prev)
    rec_m  = _mean(recent)

    improvement = prev_m - rec_m  # positive means better
    threshold  = max(ea, er * abs(prev_m))

    # also require the recent spread to be tiny (very flat)
    rec_spread = (max(recent) - min(recent)) if recent else 0.0

    plateau = (improvement <= threshold) and (rec_spread <= 2*ea)
    if plateau:
        print(f"   🛑 plateau detected: prev_mean={prev_m:.6g}, "
              f"recent_mean={rec_m:.6g}, Δ={improvement:.3g} ≤ {threshold:.3g}, "
              f"spread={rec_spread:.3g}")
    return plateau

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
                             target_score: float = 0.0095,
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

        scenario    = "fast_dip"
        var_scores: list[float] = []
        best_score  = float("inf")
        best_vec: list[float] | None = None
        best_iter   = 0

        for k in range(1, max_iter + 1):
            print(f"\n[{gname}/{var_name}] iteration {k}/{max_iter} • scenario = {scenario}")
            _enable_param_events(pf_data, bus, scenario)

            # -------- ask, evaluate, tell --------------------------------
            cand   = PSO.ask_one(gname)
            score  = _eval_candidate(pf_data, meta, bus, cand, scenario)
            PSO.tell_one(gname, score, cand)

            # record score + artifacts
            var_scores.append(float(score))
            _update_progress_artifacts(var_name, var_scores)

            # track global best (score + vector + iteration)
            if score < best_score:
                best_score = float(score)
                best_vec   = list(cand)
                best_iter  = k
                _write_best_vector_csv(var_name, best_score, best_iter, best_vec)

            # -------- convergence? ---------------------------------------
            if best_score <= target_score:
                print(f"   🎯 target achieved (best {best_score:.6g})")
                break

            if _has_plateau(var_scores):
                print(f"   ✅ steady-state fitness reached after {k} iters "
                      f"(best {best_score:.6g} @ iter {best_iter}).")
                break

        # ---------- write BEST vector back to PF & snapshot --------------
        if best_vec is not None:
            print(f"   ↩ writing best vector from iter {best_iter} (score {best_score:.6g})")
            PSO.write_candidate(pf_data, meta, best_vec)
        else:
            # fall back to optimiser’s best if never improved
            PSO.write_candidate(pf_data, meta, PSO.get_best(gname))

        _deactivate_variant(pf_data, var_name)

        # store in snapshot
        meta["AVR_Final"]   = _capture_avr_params(pf_data, meta)
        meta["final_score"] = best_score
        meta["tuned_iter"]  = best_iter
        meta["tuning_ts"]   = datetime.datetime.now().isoformat(timespec="seconds")

    snap_p.write_text(json.dumps(snap, indent=2))
    print("\n✅  Tuning finished – snapshot JSON updated")
