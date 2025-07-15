# ---------------------------------------------------------------------------
# gen_tuner.py          – minimal yet functional generator‑by‑generator
#                         tuning loop for isolated‑variant studies.
#
# * expects:
#       pf_data                       (wrapper you already use elsewhere)
#       evaluate_voltage_control()    (your scoring function, imported here)
#       PowerFactory_Control.Create_RMS_Sim  as RMS_Sim
#       PowerFactory_Control.Run_RMS_Sim     as R_RMS_Sim
#
# * snapshot JSON must contain:
#       "generators": [
#            { "name": "...", "bus": "...", "selected_for_tuning": true,
#              "Has_Trf": true/false, "trf_name": "...", "trf_hv_bus": "..." }
#       ]
# ---------------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import json
import time
import datetime
import PowerFactory_Control.Run_RMS_Sim as R_RMS_Sim
import Data_Scoring.Voltage.V_P as EV
import PowerFactory_Interaction.Tune_Isolated_Gens as TUNE
import PowerFactory_Interaction.Run_Initial_RMS as R_RMS_Sim

# ───────────────────────────────────────────────────────────────
# locate the project‑level snapshot
# ───────────────────────────────────────────────────────────────
_SNAP_BASE = (r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
              r"\Dynamic Model Builder\JSON_DB\Network_Snapshots")

def _snap_path(pf_data) -> Path:
    return Path(_SNAP_BASE) / f"{pf_data.project_name}_gen_snapshot.json"

# ───────────────────────────────────────────────────────────────
# helpers
# ----------------------------------------------------------------
def _activate_variant(pf_data, gen_name: str):
    """Activate the variant  TUNE_<GenName>.IntScheme  (creates if missing)."""
    var_id = f"TUNE_{gen_name}.IntScheme"
    try:
        scheme = pf_data.variations_folder.GetContents(var_id)[0]
    except IndexError:
        raise RuntimeError(f"Variant {var_id} not found – run the pre‑isolation step first")

    scheme.Activate()
    print(f"   ✓ variant «{scheme.loc_name}» activated")


def _enable_param_event(pf_data, bus: str):
    """Set Voltage‑Drop parameter event in‑service inside the active variant."""
    sim_name_drop = f"Voltage Drop{bus}.EvtParam"
    try:
        evt_drop = pf_data.Simulation_Folder.GetContents(sim_name_drop)[0]
    except IndexError:
        raise RuntimeError(f"Parameter event {sim_name_drop} not found – aborting")

    evt_drop.SetAttribute("e:outserv", False)

    """Set Voltage‑Drop parameter event in‑service inside the active variant."""
    sim_name_rise = f"Voltage Rise{bus}.EvtParam"
    try:
        evt_rise = pf_data.Simulation_Folder.GetContents(sim_name_rise)[0]
    except IndexError:
        raise RuntimeError(f"Parameter event {sim_name_rise} not found – aborting")

    evt_rise.SetAttribute("e:outserv", False)

    print(f"   ✓ event «{evt_rise.loc_name}» in service")
    print(f"   ✓ event «{evt_drop.loc_name}» in service")


def _run_rms_and_score(pf_data, bus: str) -> float:
    """Run RMS and return fitness score from evaluate_voltage_control()."""
    print(f"   ▶ running RMS simulation …")
    R_RMS_Sim.quick_rms_run(
        pf_data,
        "Power Flow",                       # study‑case name
        bus,
        r"C:\Users\james\OneDrive\MSc Project\results",
        None                                # no extra monitors
    )
    # a tiny pause helps PF flush the CSV on slow disks
    time.sleep(0.3)

    score, *_ = EV.evaluate_voltage_control(bus)
    print(f"   ↩ fitness = {score:.6f}")
    return score


# ───────────────────────────────────────────────────────────────
# main entry point
# ----------------------------------------------------------------
def tune_selected_generators(pf_data,
                             target_score: float = 0.00085,
                             max_iter: int = 5):
    """
    Iterate over all generators flagged  "selected_for_tuning": true  in the
    snapshot and autotune their AVR until *target_score* is met or until
    *max_iter* iterations have been tried.
    """
    snap_file = _snap_path(pf_data)
    if not snap_file.exists():
        raise FileNotFoundError(snap_file)

    snap = json.loads(snap_file.read_text())

    todo = [g for g in snap.get("generators", []) if g.get("selected_for_tuning")]
    if not todo:
        print("📭  Nothing to tune – no generator carries the flag.")
        return

    print(f"🔧  Starting tuning loop for {len(todo)} generator(s)…")

    for meta in todo:
        gname = meta["name"]
                # (*) choose which bus to monitor / fault
        if meta.get("Has_Trf"):
            bus = meta.get("Grid_Bus")  # ← prefer HV side
            if not bus:
                raise ValueError(f"Generator {gname} has Has_Trf=True but no HV bus "
                                 "given in the snapshot (fields trf_hv_bus / Grid_Bus)")
            print(f"⚙️   {gname} is transformer‑coupled – using HV bus «{bus}»")
        else:
            bus = meta["bus"]                                     # LV side = gen bus
            print(f"⚙️   {gname} directly on bus «{bus}»")

        print("\n" + "═"*65)
        print(f"⚙️   Tuning «{gname}»")


        try:
            R, X, V, U0 = R_RMS_Sim.get_bus_thevenin(pf_data, bus)       # already returns V_nom too
        except KeyError:
            raise RuntimeError(f"No Thevenin data in JSON for bus «{bus}»")

        # locate the voltage‑source object  (created earlier by Add_Voltage_Source)
        vs_name = f"{bus}V_Source.ElmVac"
        vs_list = pf_data.grid_folder.GetContents(f"{vs_name}")
        if not vs_list:
            raise RuntimeError(f"Voltage‑source object {vs_name} not found")
        V_Source_Object = vs_list[0]

        # ------------------------------------------------------------------
        # make the voltage source emulate the positive‑sequence Thevenin
        # ------------------------------------------------------------------
        ok = (
            V_Source_Object.SetAttribute("e:R1",      float(R)) == 0 and
            V_Source_Object.SetAttribute("e:X1",      float(X)) == 0 and
            V_Source_Object.SetAttribute("e:vtarget", U0        ) == 0     # pu value
        )
        if not ok:
            raise RuntimeError(f"Could not set R/X/vtarget on {vs_name}")

        print(f"⚙️   {vs_name}:  R1={R:.4f} Ω  X1={X:.4f} Ω  vtarget={U0:.4f} pu")

        # 1) activate isolated‑variant
        _activate_variant(pf_data, gname)

        # 2) enable the parameter event in this variant
        _enable_param_event(pf_data, bus)

        # 3) iterative RMS → score → (optional) retune loop
        for k in range(1, max_iter+1):
            score = _run_rms_and_score(pf_data, bus)

            if score <= target_score:
                print(f"   🎉 target reached in {k} iteration(s)\n")
                break

            changed = TUNE._tune_avr_parameters(pf_data, meta, k, score)
            if not changed:
                print(f"   ⏹ stopping – AVR unchanged or custom tuner gave up")
                break

        # you might wish to store final score inside the JSON:
        meta["final_score"]    = score
        meta["tuning_done_ts"] = datetime.now().isoformat(timespec="seconds")

    # optionally persist the updated snapshot
    snap_file.write_text(json.dumps(snap, indent=2))
    print("\n✅  Tuning complete – snapshot updated.")




