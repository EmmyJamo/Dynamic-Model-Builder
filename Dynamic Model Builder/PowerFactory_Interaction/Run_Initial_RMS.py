import PowerFactory_Control.Create_RMS_Sim as RMS_Sim
import PowerFactory_Control.Run_RMS_Sim as R_RMS_Sim
from pathlib import Path
from typing import Dict, Tuple, Union
import json

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


def build_simulation_and_run(pf_data, Voltage_Set_Point_Drop, Voltage_Drop_Time, Voltage_Rise_Time):

    """
    Build a simulation event for a voltage drop at a specific bus.
    """

    # ----- 0. obtain the bus list -------------------------------------------
    buses = get_target_buses(pf_data)

    for bus in buses:
        print(f"\n🔄  Processing bus «{bus}»")

        R, X, V_kV, U0 = get_bus_thevenin(pf_data, bus)

        # (a) Add / configure the AC voltage source
        RMS_Sim.Add_Voltage_Source(
            pf_data,
            Busbar  = bus,
            voltage = V_kV,        # ⚙️ use Unom in p.u.; change if needed
            R       = R,        # ⚙️ or use Rth/Xth from JSON if wanted
            X       = X, 
            U0      = U0,
            Isolated_System_TF = False # Isolated_System_TF
        )

        # (b) Create the simulation event for voltage drop
        RMS_Sim.Create_Simulation_Event(pf_data, 
                                        'Voltage Drop' + bus, 
                                        Voltage_Drop_Time, Voltage_Set_Point_Drop, 
                                        bus + 'V_Source.ElmVac')  


        # (b.1) Create the simulation event for voltage rise
        RMS_Sim.Create_Simulation_Event(pf_data, 
                                        'Voltage Rise' + bus, 
                                        Voltage_Rise_Time, U0, 
                                        bus + 'V_Source.ElmVac')  

        # (c) Create the results variable for the bus
        RMS_Sim.Create_Results_Variable(pf_data, bus)

        R_RMS_Sim.quick_rms_run(pf_data, 'Power Flow', bus, 
                                r'C:\Users\james\OneDrive\MSc Project\results', None)

        Simulation_Object = pf_data.Simulation_Folder.GetContents('Voltage Drop' + bus)[0]

        Simulation_Object.SetAttribute('e:outserv', True)  # Turning all of these off at first to be turned on when needed - wont have to cycle through and turn off each one, for each sim\

        Simulation_Object = pf_data.Simulation_Folder.GetContents('Voltage Rise' + bus)[0]

        Simulation_Object.SetAttribute('e:outserv', True)  # Turning all of these off at first to be turned on when needed - wont have to cycle through and turn off each one, for each sim\

        V_Source_Object = pf_data.grid_folder.GetContents(bus + 'V_Source.ElmVac')[0]

        V_Source_Object.SetAttribute('e:outserv', True)  # Turning all of these off at first to be turned on when needed - wont have to cycle through and turn off each one, for each sim


