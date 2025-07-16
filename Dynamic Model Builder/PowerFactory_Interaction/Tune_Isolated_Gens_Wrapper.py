# ---------------------------------------------------------------------------
# gen_tuner.py – generator-by-generator AVR tuning loop
# ---------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
import json
import time
import datetime
import traceback

# --- PF / project-specific imports -----------------------------------------
import PowerFactory_Control.Run_RMS_Sim             as R_RMS_Sim
import Data_Scoring.Voltage.V_P                    as EV
import PowerFactory_Interaction.Tune_Isolated_Gens as TUNE

# ---------------------------------------------------------------------------
# snapshot path helper
# ---------------------------------------------------------------------------
_SNAP_BASE = (r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
              r"\Dynamic Model Builder\JSON_DB\Network_Snapshots")

def _snap_path(pf_data) -> Path:
    return Path(_SNAP_BASE) / f"{pf_data.project_name}_gen_snapshot.json"


# ---------------------------------------------------------------------------
# AVR capture helper
# ---------------------------------------------------------------------------
_AVR_TAGS = (
    "Tr", "Ka", "Ta", "Vrmax", "Vrmin", "Ke", "Te",
    "Kf", "Tf", "E1", "E2", "Se1", "Se2"
)

def _find_avr_block(pf_data, meta: dict):
    """
    Locate the AVR ElmDsl object for the generator described by *meta*.
    Uses meta["AVR_Name"] if present; otherwise searches under the plant /
    composite model (meta['name']) for *avr* in the block name.
    Returns the block or None.
    """
    avr_name = meta.get("AVR_Name")
    if avr_name:
        # Most robust: scan whole project for matching ElmDsl name
        blocks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
        for blk in blocks:
            if blk.loc_name == avr_name:
                return blk

    # fallback: look under the generator's plant model if available
    try:
        gen_obj = pf_data.app.GetCalcRelevantObjects(meta["name"] + ".*")[0]
        for blk in gen_obj.GetContents("*.ElmDsl"):
            if "avr" in blk.loc_name.lower():
                return blk
    except Exception:
        pass

    # last resort: global search by substring 'avr'
    blocks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
    for blk in blocks:
        if "avr" in blk.loc_name.lower():
            # If multiple, return the first (caller warned in debug print)
            return blk
    return None


def _capture_avr_params(pf_data, meta: dict) -> dict:
    """
    Read the core AVR parameters from the AVR ElmDsl block.
    Missing params are skipped. Returns {tag: float, ...}.
    """
    params = {}
    blk = _find_avr_block(pf_data, meta)
    if blk is None:
        print(f"   ⚠️  AVR block not found for {meta.get('name')}")
        return params

    for tag in _AVR_TAGS:
        try:
            val = blk.GetAttribute(f"e:{tag}")  # most PF tunables are 'e:' (edit)
            if val is None:                     # if not there try plain name
                val = blk.GetAttribute(tag)
            if val is not None:
                params[tag] = float(val)
        except Exception:
            # ignore silently? no: emit debug
            print(f"      ↪ param {tag} unavailable on {blk.loc_name}")
    return params


# ---------------------------------------------------------------------------
# helpers – PF actions
# ---------------------------------------------------------------------------
def _activate_variant(pf_data, gen_name: str) -> None:
    """Activate existing variant  TUNE_<GenName>.IntScheme."""
    var_id = f"TUNE_{gen_name}.IntScheme"
    try:
        scheme = pf_data.variations_folder.GetContents(var_id)[0]
    except IndexError as e:
        raise RuntimeError(f"Variant {var_id} not found – run isolation step") from e

    scheme.Activate()
    print(f"   ✓ variant «{scheme.loc_name}» activated")


def _enable_param_events(pf_data, bus: str) -> None:
    """Switch ON voltage-dip & rise events for *bus* inside current variant."""
    for label in ("Drop", "Rise"):
        evt_name = f"Voltage {label}{bus}.EvtParam"
        try:
            evt = pf_data.Simulation_Folder.GetContents(evt_name)[0]
            evt.SetAttribute("e:outserv", False)
            print(f"   ✓ event «{evt.loc_name}» in service")
        except IndexError:
            raise RuntimeError(f"Parameter event {evt_name} not found")


def _set_voltage_source(pf_data, bus: str, R: float, X: float, U0: float) -> None:
    """Push R, X, vtarget into  <bus>V_Source.ElmVac  (must exist)."""
    vs_name = f"{bus}V_Source.ElmVac"
    vs_list = pf_data.grid_folder.GetContents(vs_name)
    if not vs_list:
        raise RuntimeError(f"Voltage-source object {vs_name} not found")

    vs = vs_list[0]
    ok = (
        vs.SetAttribute("e:R1",      float(R)) == 0 and
        vs.SetAttribute("e:X1",      float(X)) == 0 and
        vs.SetAttribute("e:vtarget", float(U0)) == 0
    )
    if not ok:
        raise RuntimeError(f"Setting R/X/vtarget failed on {vs_name}")

    print(f"⚙️   {vs_name}:  R1={R:.4f} Ω  X1={X:.4f} Ω  vtarget={U0:.4f} pu")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def tune_selected_generators(pf_data,
                             target_score: float = 8.5e-4,
                             max_iter: int = 5,
                             dry_run: bool = False) -> None:
    """
    Iterate over all generators flagged `"selected_for_tuning": true`
    and autotune their AVR until *target_score* is met or *max_iter*
    iterations are exceeded.

    *dry_run* = True ⇒ skip PF calls & scoring (for logic testing).
    """
    snap_file = _snap_path(pf_data)
    if not snap_file.exists():
        raise FileNotFoundError(snap_file)

    snap = json.loads(snap_file.read_text())
    todo = [g for g in snap.get("generators", []) if g.get("selected_for_tuning")]

    if not todo:
        print("📭  Nothing to tune – no generator carries the flag.")
        return

    print(f"🔧  Tuning loop for {len(todo)} generator(s) – target score {target_score}")

    # ------------------------------------------------------------------ loop
    for meta in todo:
        gname = meta["name"]

        # decide which bus we fault / monitor
        if meta.get("Has_Trf"):
            bus = meta.get("Grid_Bus")
            if not bus:
                raise RuntimeError(f"{gname}: Has_Trf == true but Grid_Bus missing")
            print(f"\n🗲  {gname} (trf-coupled) → HV bus «{bus}»")
        else:
            bus = meta["bus"]
            print(f"\n🗲  {gname} on LV bus «{bus}»")

        print("═" * 60)

        try:
            R, X, _, U0 = R_RMS_Sim.get_bus_thevenin(pf_data, bus)
        except Exception as e:
            print(f"⚠️  Thevenin lookup failed for {bus}: {e}")
            traceback.print_exc()
            continue

        try:
            if not dry_run:
                _activate_variant(pf_data, gname)
                _enable_param_events(pf_data, bus)
                _set_voltage_source(pf_data, bus, R, X, U0)

            print(f"   🔧 seeding AVR parameters …")
            if not dry_run:
                TUNE._seed_avr_parameters(pf_data, meta, bus)
        except Exception as e:
            print(f"⚠️  setup failed for {gname}: {e}")
            traceback.print_exc()
            continue

        # ------------------------------------------------ iterative tuning
        last_score = float("inf")
        iters_done = 0
        for k in range(1, max_iter + 1):
            iters_done = k
            print(f"\n   — iteration {k}/{max_iter} —")

            # ---- run RMS & score -----------------------------------------
            if dry_run:
                last_score = target_score * (2 / k)        # fake improving
                print(f"   [dry] fake score = {last_score:.6f}")
            else:
                try:
                    R_RMS_Sim.quick_rms_run(
                        pf_data,
                        "Power Flow",
                        bus,
                        r"C:\Users\james\OneDrive\MSc Project\results",
                        None
                    )
                except Exception as e:
                    print(f"⚠️  RMS run failed: {e}")
                    traceback.print_exc()
                    break

                time.sleep(0.3)            # let PF flush CSV

                try:
                    last_score, *_ = EV.evaluate_voltage_control(bus)
                    print(f"   ↩ fitness = {last_score:.6f}")
                except Exception as e:
                    print(f"⚠️  scoring failed: {e}")
                    traceback.print_exc()
                    break

            # ---- convergence? -------------------------------------------
            if last_score <= target_score:
                print(f"   🎉 target achieved (score {last_score:.6f})")
                # capture AVR params at success
                try:
                    avr_vals = _capture_avr_params(pf_data, meta) if not dry_run else {}
                    meta["AVR_Final"] = avr_vals
                    print(f"   📥 saved AVR params: {avr_vals}")
                except Exception as e:
                    print(f"⚠️  Could not capture AVR params: {e}")
                    traceback.print_exc()
                break

            # ---- ML / PSO+CMA-ES parameter update -----------------------
            try:
                changed = TUNE._tune_avr_parameters(
                    pf_data, meta, iteration=k, current_score=last_score, dry_run=dry_run
                )
            except Exception as e:
                print(f"⚠️  tuner crashed: {e}")
                traceback.print_exc()
                break

            if not changed:
                print("   ⏹ tuner reports no further change; stopping.")
                break

        # store outcome in snapshot (always)
        meta["final_score"]        = last_score
        meta["tuned_iter"]         = iters_done
        meta["tuning_timestamp"]   = datetime.datetime.now().isoformat(timespec="seconds")
        # If we exited loop without meeting target and never captured params,
        # capture them now (best-effort) so snapshot shows what we ended with:
        if "AVR_Final" not in meta and not dry_run:
            try:
                avr_vals = _capture_avr_params(pf_data, meta)
                meta["AVR_Final"] = avr_vals
                print(f"   📥 (final) saved AVR params after loop: {avr_vals}")
            except Exception as e:
                print(f"⚠️  Could not capture final AVR params: {e}")
                traceback.print_exc()

    # ---------------------------------------------------------------- persist
    try:
        snap_file.write_text(json.dumps(snap, indent=2))
        print(f"\n✅  Tuning finished – snapshot updated → {snap_file}")
    except Exception as e:
        print(f"⚠️  Could not update snapshot JSON: {e}")
