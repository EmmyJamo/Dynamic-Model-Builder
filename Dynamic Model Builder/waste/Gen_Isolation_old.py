# ---------------------------------------------------------------------------
# pf_island_tools.py   –  isolate selected generators on infinite buses
# ---------------------------------------------------------------------------
import json, traceback
import math
from pathlib import Path
SNAP_BASE = (r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
             r"\Dynamic Model Builder\JSON_DB\Network_Snapshots")

# ───────────────── helpers ────────────────────────────────────────────────
def _snap_path(pf_Data):
    return Path(SNAP_BASE) / f"{pf_Data.project_name}_gen_snapshot.json"


# ───────────────── main routine ───────────────────────────────────────────
def build_infinite_bus_islands(pf_data):
    """Isolate every generator whose JSON flag `selected_for_tuning` is true."""
    try:
        snap = json.loads(_snap_path(pf_data).read_text())
    except Exception as e:
        raise RuntimeError(f"Snapshot JSON unreadable – {e}")

    todo = [g for g in snap["generators"] if g.get("selected_for_tuning")]
    if not todo:
        print("No generators marked → nothing to isolate."); return


    for meta in todo:
        gname   = meta["name"]
        # accept any of the four possible key spellings
        r_th    = meta.get("Rth_ohm")
        x_th    = meta.get("Xth_ohm")
        kv      = meta.get("voltage")
        phiui = float(meta.get("phiui"))  # angle in degrees
        #XtoR = meta.get("XtoR")  # X/R ratio
        RtoX = meta.get("RtoX")  # R/X ratio
        #XtoR_b = meta.get("XtoR_b")  # X/R ratio (base)
        #RtoX_b = meta.get("RtoX_b")  # R/X ratio (base)
        Skss = meta.get("Skss_MVA")  # short-circuit power in MVA
        Ikss = meta.get("Ikss_kA")  # short-circuit current in kA
        #ip = meta.get("ip_A")  # short-circuit current in A
        pow_fac = float(math.cos((math.radians(phiui))))

        type_gen = meta.get("type")

        if None in (r_th, x_th, kv):
            print(f"⚠️  {gname}: missing Zth data → skipped"); 

        print("Starting transformer creation process.")

        # Create Terminal for the Generator
        terminal_name = gname + '_BB'
        terminal = pf_data.project.CreateObject('ElmTerm', terminal_name)
        if not terminal:
            raise Exception(f"Failed to create terminal for generator '{gname}'")
        print(f"Terminal '{terminal_name}' created successfully.")

        # Create the Infbus
        infbus_name = gname + '_IB'
        infbus = pf_data.project.CreateObject('ElmXnet', infbus_name)
        if not infbus:
            raise Exception(f"Failed to create transformer '{infbus_name}'")
        print(f"Transformer '{infbus_name}' created successfully.")

        # Move the Terminal and Infbus to the Grid Folder
        pf_data.grid_folder.Move(terminal)
        pf_data.grid_folder.Move(infbus)
        print(f"Moved terminal and transformer to grid folder '{pf_data.grid_folder}'.")
            
        terminal.SetAttribute('e:uknom', kv)          

        # Connect infbus to Generator's Terminal (via cubicle)
        generator_cubicle = terminal.CreateObject('StaCubic', infbus_name + '_C')
        if not generator_cubicle:
            raise Exception(f"Failed to create cubicle for terminal '{terminal_name}'")
        print(f"Generator cubicle '{infbus_name + '_C'}' created successfully.")

        infbus.SetAttribute('e:bus1', generator_cubicle)  # High voltage side (generator cubicle)
        print(f"Infbus low voltage side connected to generator cubicle '{infbus_name + '_C'}'.")

        # Connect Generator to the transformer
        generator_cubicle_2 = terminal.CreateObject('StaCubic', infbus_name + '_C2')
        if not generator_cubicle_2:
            raise Exception(f"Failed to create cubicle for terminal '{terminal_name}'")
        print(f"Generator cubicle 2 '{infbus_name + '_C2'}' created successfully.")
            
        if type_gen == "synchronous": 
            generator = pf_data.grid_folder.GetContents(gname + ".ElmSym")[0]
        elif type_gen == "inverter":
            generator = pf_data.grid_folder.GetContents(gname + ".ElmGenstat")[0]
        generator.SetAttribute('e:bus1', generator_cubicle_2)

        print(f"Generator '{gname}' connected to cubicle 2 '{infbus_name + '_C2'}'.")

        infbus.SetAttribute('e:bustp', "SL")  # SL = Slack bus
        infbus.SetAttribute('e:cused', 0) # 0 = max values 1 = min values
        infbus.SetAttribute('e:snss', Skss)
        infbus.SetAttribute('e:ikss', Ikss)
        infbus.SetAttribute('e:rntxn', RtoX)
        infbus.SetAttribute('e:cosn', pow_fac)

        print(f"✅  {gname} isolated on IB  (Zth={r_th:.4f}+j{x_th:.4f} Ω)")

    print(f"\n🏁  Isolation done generator(s) moved.")




