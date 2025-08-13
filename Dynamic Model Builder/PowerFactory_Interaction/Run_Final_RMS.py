# ─────────────────────────────────────────────────────────────────────────────
# Final re-run & scoring of all buses (after tuning) → write into buses[]
# Seeds AVR with the ML-tuned params from generators[*]["AVR_Final"].
# ─────────────────────────────────────────────────────────────────────────────
import time, json, datetime
from pathlib import Path

import Data_Scoring.Voltage.V_P as EV
import PowerFactory_Control.Run_RMS_Sim as R_RMS_Sim
import PowerFactory_Interaction.Run_Initial_RMS as R_Init_RMS
from PowerFactory_Interaction.Add_Seed_AVR_Values import _seed_avr_parameters

_FINAL_RESULTS_ROOT = Path(r"C:\Users\james\OneDrive\MSc Project")

_SCEN_MAP = {
    "fast_dip": {
        "rise_evt": "Voltage Rise 0.2s",
        "results_dir": _FINAL_RESULTS_ROOT / "results_2.2_rise",
        "json_key": "fitness_fast_dip_final",
    },
    "slow_hold": {
        "rise_evt": "Voltage Rise 7s",
        "results_dir": _FINAL_RESULTS_ROOT / "results_7_rise",
        "json_key": "fitness_slow_hold_final",
    },
}

def _pf_get(pf_data, folder_attr: str, name: str):
    try:
        lst = getattr(pf_data, folder_attr).GetContents(name)
        return lst[0] if lst else None
    except Exception:
        return None

def _set_outserv(obj, outserv: bool):
    if obj is not None:
        try:
            obj.SetAttribute("e:outserv", outserv)
        except Exception:
            pass

def run_final_full_simulations(pf_data, *, pause_s: float = 0.25) -> None:
    """
    For every target bus:
      1) If a generator is attached, seed its AVR with generators[*]["AVR_Final"].
      2) Re-run both scenarios (fast_dip, slow_hold).
      3) Score each run and write scores back to the bus entry in the snapshot
         as 'fitness_fast_dip_final' and 'fitness_slow_hold_final'.
    """
    # ensure output dirs exist
    for s in _SCEN_MAP.values():
        Path(s["results_dir"]).mkdir(parents=True, exist_ok=True)

    # load snapshot + lookups
    snap_path = R_Init_RMS._snap_path(pf_data)
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    buses = snap.get("buses", [])
    gens  = snap.get("generators", [])

    name_to_bus_idx = {b.get("name"): i for i, b in enumerate(buses)}
    gen_by_name     = {g.get("name"): g for g in gens}

    buses_info = R_Init_RMS.annotate_target_buses_with_gens(pf_data)
    print(f"\n🧪 Re-running final simulations on {len(buses_info)} bus(es)…")

    for bus, has_gen, gen_name in buses_info:
        print(f"\n— Bus «{bus}» —")

        # 1) Seed AVR from generators[*]["AVR_Final"] (preferred)
        if has_gen and gen_name in gen_by_name:
            meta = gen_by_name[gen_name]
            json_key = None
            if "AVR_Final" in meta and isinstance(meta["AVR_Final"], dict):
                json_key = "AVR_Final"
            elif "AVR_FinalVec" in meta and isinstance(meta["AVR_FinalVec"], dict):
                json_key = "AVR_FinalVec"
            elif "AVR_Seed" in meta and isinstance(meta["AVR_Seed"], dict):
                json_key = "AVR_Seed"

            if json_key:
                try:
                    _seed_avr_parameters(pf_data, meta, gen_name, json_key=json_key, dry_run=False)
                    print(f"  ✓ AVR seeded for «{gen_name}» from '{json_key}'")
                except Exception as e:
                    print(f"  ⚠️ AVR seeding failed for «{gen_name}»: {e}")
            else:
                print(f"  ⚠️ no AVR params found in snapshot for «{gen_name}» (skipping seeding)")
        elif has_gen:
            print(f"  ⚠️ generator meta not found in snapshot for «{gen_name}» (skipping seeding)")

        # PF objects needed to run the scenarios
        vac       = _pf_get(pf_data, "grid_folder",       f"{bus}V_Source.ElmVac")
        drop_evt  = _pf_get(pf_data, "Simulation_Folder", f"Voltage Drop {bus}")
        rise_fast = _pf_get(pf_data, "Simulation_Folder", f"{_SCEN_MAP['fast_dip']['rise_evt']} {bus}")
        rise_slow = _pf_get(pf_data, "Simulation_Folder", f"{_SCEN_MAP['slow_hold']['rise_evt']} {bus}")

        if vac is None or drop_evt is None or rise_fast is None or rise_slow is None:
            print("  ⚠️  Missing Vac/Drop/Rise event(s) — skipping this bus.")
            continue

        bus_idx = name_to_bus_idx.get(bus)

        # enable Vac while running
        _set_outserv(vac, False)

        # 2) run & score both scenarios
        for tag in ("fast_dip", "slow_hold"):
            info = _SCEN_MAP[tag]
            results_dir = Path(info["results_dir"])

            # activate proper events
            if tag == "fast_dip":
                _set_outserv(rise_fast, False)
                _set_outserv(rise_slow, True)
            else:
                _set_outserv(rise_fast, True)
                _set_outserv(rise_slow, False)
            _set_outserv(drop_evt, False)

            print(f"  ▶ {tag}: RMS → {results_dir}")
            R_RMS_Sim.quick_rms_run(pf_data, "Power Flow", bus, str(results_dir), None)

            time.sleep(pause_s)

            # score and persist
            try:
                score = float(EV.evaluate_voltage_control(bus, results_dir))
                print(f"  ✓ {tag} score = {score:.6g}")
            except Exception as e:
                print(f"  ❌ scoring failed for {tag} on {bus}: {e}")
                score = float("nan")

            if bus_idx is not None:
                buses[bus_idx][info["json_key"]] = score

            # reset PF and deactivate this rise
            try:
                pf_data.app.ResetCalculation()
            except Exception:
                pass
            if tag == "fast_dip":
                _set_outserv(rise_fast, True)
            else:
                _set_outserv(rise_slow, True)

        # disable drop & Vac before moving on
        _set_outserv(drop_evt, True)
        _set_outserv(vac, True)

    # 3) write back snapshot
    snap["buses"] = buses
    snap["final_full_scores_ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"\n📄 Final scores written to {snap_path} "
          f"(keys: {_SCEN_MAP['fast_dip']['json_key']}, {_SCEN_MAP['slow_hold']['json_key']}).")
