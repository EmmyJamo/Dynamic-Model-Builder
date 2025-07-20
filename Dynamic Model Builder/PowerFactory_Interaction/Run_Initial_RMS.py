from ast import Raise
import PowerFactory_Control.Create_RMS_Sim as RMS_Sim
import PowerFactory_Control.Run_RMS_Sim as R_RMS_Sim
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import json
from typing import List, Tuple
import traceback
from pathlib import Path
from typing import Iterable

SNAP_BASE = (r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
             r"\Dynamic Model Builder\JSON_DB\Network_Snapshots")

def _snap_path(pf_data):
    return Path(SNAP_BASE) / f"{pf_data.project_name}_gen_snapshot.json"

def get_bus_thevenin(pf_data, bus_name: str) -> tuple[float, float, float]:
    """
    Return (R_ohm, X_ohm, V_nom_kV) for a single bus.

    Raises
    ------
    KeyError  if the bus is not in the snapshot JSON.
    """
    rxv_map = load_bus_thevenin_map(pf_data, with_voltage=True)   # {bus:(R,X,V)}
    try:
        return rxv_map[bus_name]          # unpack later →  R, X, V = ...
    except KeyError:
        raise KeyError(f"Bus '{bus_name}' not found in snapshot JSON")

def load_bus_thevenin_map(
    pf_data,
    *,
    with_voltage: bool = False           # ← set True if you need V_nom_kV
) -> Dict[str, Union[Tuple[float, float],
                     Tuple[float, float, float]]]:
    """
    Return a dict whose keys are bus names.

        • Default  (with_voltage = False):
              { "Bus 1": (R_ohm, X_ohm), ... }

        • with_voltage = True:
              { "Bus 1": (R_ohm, X_ohm, V_nom_kV), ... }

    Works with either snapshot layout:
      • "buses"  array (preferred – produced by add_bus_thevenin_to_snapshot)
      • or  R/X values embedded in each generator record (legacy fallback)

    Missing numbers are returned as 0.0.
    """
    snap_path = _snap_path(pf_data)
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_path}")

    with open(snap_path, encoding="utf-8") as fp:
        snap = json.load(fp)
    z_by_bus: dict[str, tuple] = {}

    # ───── preferred layout ────────────────────────────────────────────
    if "buses" in snap:
        for b in snap["buses"]:
            R = b.get("Rth_ohm", 0.0)
            X = b.get("Xth_ohm", 0.0)
            V = b.get("V_nom_kV", 0.0)
            U0 = b.get("U0_pu", 1.0)  # default to 1.0 p.u. if not present
            z_by_bus[b["name"]] = (R, X, V, U0) if with_voltage else (R, X)

    # ───── legacy fallback (values sit inside generator entries) ───────
    elif "generators" in snap:
        for g in snap["generators"]:
            if "Rth_ohm" in g and "Xth_ohm" in g:
                R = g["Rth_ohm"]
                X = g["Xth_ohm"]
                V = 0.0                         # no voltage info available
                z_by_bus[g["bus"]] = (R, X, V) if with_voltage else (R, X)

    return z_by_bus

BusInfo = Tuple[str, bool, Optional[str]]   # (bus, has_gen?, gen_name or None)

def annotate_target_buses_with_gens(pf_data) -> List[BusInfo]:
    """
    1)  Uses your existing get_target_buses() to get the list of buses
        that will receive voltage‑step simulations.
    2)  For each of those buses, checks the generator table:
        • If the bus appears as  g["Grid_Bus"]    (HV side of xfmr)  OR
                                 g["bus"]         (direct LV side)
          → marks has_gen=True and returns that generator's name.
        • Otherwise has_gen=False and gen_name=None.
    """
    # --- A. get the target bus list you already generate --------------------
    target_buses = get_target_buses(pf_data)         # <- returns List[str]

    # --- B. read snapshot JSON once ----------------------------------------
    snap_path = _snap_path(pf_data)
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_path}")
    snap = json.loads(Path(snap_path).read_text(encoding="utf-8"))

    gens = snap.get("generators", [])

    # --- C. build quick lookup  bus -> generator name -----------------------
    bus2gen: dict[str, str] = {}
    for g in gens:
        # direct‑connected bus
        lv_bus = g.get("bus")
        if lv_bus:
            bus2gen.setdefault(lv_bus, g["name"])

        # HV bus of xfmr‑coupled unit
        hv_bus = g.get("Grid_Bus")
        if hv_bus:
            bus2gen.setdefault(hv_bus, g["name"])

    # --- D. assemble annotated list ----------------------------------------
    annotated: List[BusInfo] = []
    for bus in target_buses:
        gen_name = bus2gen.get(bus)        # None if no generator on that bus
        has_gen  = gen_name is not None
        annotated.append((bus, has_gen, gen_name))

    return annotated
# ───────────────────────────────────────────────────────────────
# helper: return only “good” buses for voltage-source placement
# ───────────────────────────────────────────────────────────────
def get_target_buses(pf_data) -> list[str]:
    """
    Read the snapshot JSON and build a list of buses **without** a
    transformer-coupled generator.

    Returns
    -------
    list[str]
        Ordered list of bus `loc_name`s on which you should add
        AC-voltage sources / parameter events.

    JSON assumptions
    ----------------
    * Top-level key  "buses": [{ "name": "<BusName>", ... }, …]
    * Top-level key  "generators": each generator entry contains
        - "bus"       : the LV bus where its cubicle is connected
        - "Has_Trf"   : boolean  (True ⇢ gen sits behind a transformer)
    """
    BusGenTupleList  = List[Tuple[str, Union[str, None]]]

    snap_path = _snap_path(pf_data)
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_path}")

    with open(snap_path, encoding="utf-8") as fp:
        snap = json.load(fp)

    # --- 1. full set of buses present in the system --------------------------
    all_bus_names = [b["name"] for b in snap.get("buses", [])]

    # --- 2. collect buses to *exclude* (gens behind transformers) ------------
    exclude = {
        g["bus"]
        for g in snap.get("generators", [])
        if g.get("Has_Trf")      # only if flag exists and is True
    }

    # --- 3. build the final list ---------------------------------------------
    target = [bus for bus in all_bus_names if bus not in exclude]

    print(f"⚙️  Found {len(all_bus_names)} total buses")
    print(f"   ↪ {len(exclude)} excluded (LV side of generator transformer)")
    print(f"   → {len(target)} buses left for simulations")

    return target


def _debug(msg: str):
    print(f"[DEBUG] {msg}")

def _safe_pf_call(func, *args, **kwargs):
    """Wrapper: call PowerFactory API safely and print traceback on error."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] {func.__name__} failed → {e}")
        traceback.print_exc()     # full stack trace to console
        return None               # allows script to continue

# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------
def build_simulation_and_run(pf_data, SCENARIOS: Iterable[tuple]):
    """
    For every target LV bus:
        • add a Vac source and two parameter‑events (fast_dip & slow_hold)
        • set result channels
        • run RMS simulations
    """

    # ---------- 0. validate SCENARIOS --------------------------------------
    scenarios_checked = []
    for no, s in enumerate(SCENARIOS, start=1):
        if not isinstance(s, (list, tuple)) or len(s) != 4:
            raise ValueError(
                f"SCENARIOS item #{no} must be a 4‑tuple "
                "(tag, drop_lvl, t_drop, t_rise) – got {s!r}"
            )
        scenarios_checked.append(tuple(s))   # ensure tuple type

    _debug(f"{len(scenarios_checked)} scenario(s) validated")

    # ---------- 1. obtain bus list (+ generator info) ----------------------
    try:
        list_of_bus_gen = annotate_target_buses_with_gens(pf_data)
    except Exception as e:
        raise RuntimeError("annotate_target_buses_with_gens() failed") from e

    _debug(f"{len(list_of_bus_gen)} buses to process")

    # ---------- 2. main loop over buses ------------------------------------
    for bus, has_gen, gen_name in list_of_bus_gen:
        print(f"\n🔄  Processing bus «{bus}»"
              f"   (generator: {gen_name if has_gen else 'none'})")

        # 2‑A  thevenin lookup
        try:
            R, X, V_kV, U0 = get_bus_thevenin(pf_data, bus)
        except KeyError as e:
            print(f"[SKIP] {e}")
            continue

        # 2‑B  add / configure Vac source
        _safe_pf_call(
            RMS_Sim.Add_Voltage_Source,
            pf_data,
            Busbar=bus,
            voltage=V_kV,
            R=R,
            X=X,
            U0=U0,
            Isolated_System_TF=False
        )

        # ---------- 3. inner loop over scenarios ---------------------------
        for tag, drop_lvl, t_drop, t_rise in scenarios_checked:
            print(f"\n=== Scenario «{tag}» on {bus} ===")

            # names for parameter events
            rise_name = {
                'fast_dip': "Voltage Rise 0.2s",
                'slow_hold': "Voltage Rise 7s"
            }.get(tag)

            if rise_name is None:
                print(f"[WARN] Unknown scenario tag «{tag}» – skipping")
                continue

            drop_name = f"Voltage Drop {bus}"
            rise_name_full = f"{rise_name} {bus}"

            # (a) create / reuse drop
            if tag == "fast_dip":   # create only once
                _safe_pf_call(
                    RMS_Sim.Create_Simulation_Event,
                    pf_data, drop_name, t_drop, drop_lvl,
                    f"{bus}V_Source.ElmVac"
                )
                            # (b) set result channels
                _safe_pf_call(RMS_Sim.Create_Results_Variable, pf_data, bus)
                if has_gen:
                    _safe_pf_call(RMS_Sim.Create_Results_Variable, pf_data, gen_name)


            # (a.1) create rise event
            _safe_pf_call(
                RMS_Sim.Create_Simulation_Event,
                pf_data, rise_name_full, t_rise, U0,
                f"{bus}V_Source.ElmVac"
            )

            # (c) run RMS
            results_dir = (
                r"C:\Users\james\OneDrive\MSc Project\results_2.2_rise"
                if tag == "fast_dip" else
                r"C:\Users\james\OneDrive\MSc Project\results_7_rise"
            )
            _safe_pf_call(
                R_RMS_Sim.quick_rms_run,
                pf_data, "Power Flow", bus, results_dir, None
            )
            # NEW ─ reset PF’s internal simulation state
            _safe_pf_call(pf_data.app.ResetCalculation)

            # (d) deactivate rise event
            _safe_pf_call(
                lambda s_obj: s_obj.SetAttribute("e:outserv", True),
                pf_data.Simulation_Folder.GetContents(rise_name_full)[0]
            )
        
        # ---------- 4. deactivate extras on this bus ------------------------
        _safe_pf_call(
            lambda vac: vac.SetAttribute("e:outserv", True),
            pf_data.grid_folder.GetContents(f"{bus}V_Source.ElmVac")[0]
        )
        _safe_pf_call(
            lambda s_obj: s_obj.SetAttribute("e:outserv", True),
            pf_data.Simulation_Folder.GetContents(f"Voltage Drop {bus}")[0]
        )

        _debug(f"Finished bus «{bus}»")

    print("\n✔️  All requested simulations attempted. Check log for errors.")






