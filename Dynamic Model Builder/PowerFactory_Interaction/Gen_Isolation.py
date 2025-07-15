# ---------------------------------------------------------------------------
# pf_island_tools.py  –  isolate selected generators on infinite buses
#                         one network-variant (IntScheme) per generator
#                         (no need to keep track of a base variant)
# ---------------------------------------------------------------------------
import json, traceback, math
from pathlib import Path
import PowerFactory_Control.Get_Nested_Folder as GetFolder

SNAP_BASE = (r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
             r"\Dynamic Model Builder\JSON_DB\Network_Snapshots")


# ───────────────── helpers ────────────────────────────────────────────────
def _snap_path(pf_data):
    return Path(SNAP_BASE) / f"{pf_data.project_name}_gen_snapshot.json"


def _get_or_make_variant(pf_data, var_name):
    """
    Return an existing IntScheme called *var_name* or create a NEW one.
    Creating an empty IntScheme automatically inherits the base network.
    """
    pf_data.variations_folder = GetFolder.get_nested_folder(pf_data, ['Network Model', 'Network Data', 'Variations.IntPrjfolder'])

    if var_name in pf_data.variations_folder.GetContents('*.IntScheme'):
        print(f"➖  variant «{var_name}» already exists, using it")
        return pf_data.ariations_folder.GetContents(var_name)[0]
    else:
        print(f"➕  creating new variant «{var_name}»")
        new_var = pf_data.variations_folder.CreateObject('IntScheme', var_name)
        new_var.Activate()
        print(f"Var activated" + var_name)

        #var_name_folder = variations_folder.GetContents(var_name)[0]

        expantions_folder = GetFolder.get_nested_folder(pf_data, ['Network Model', 'Network Data', 'Variations', var_name])
        
        new_expansion = expantions_folder.CreateObject('IntSstage', var_name + '_Exp')
        new_expansion.Activate()

        print(f"➕  created network-variant «{var_name}»")
        return new_var, new_expansion


# ------------------------------------------------------------------
# helper: build the set of loc_names that must survive
# ------------------------------------------------------------------
def _keep_set(meta) -> set[str]:
    keep = {
        meta["name"],                 # the generator
        meta["bus"],                  # its LV bus
        f"{meta['bus']}V_Source",     # LV‑side Vac source
    }

    # transformer + its HV bus (if they exist in the JSON)
    trf_name     = meta.get("Trf_Name")
    trf_hv_bus   = meta.get("Grid_Bus")

    if trf_name:
        keep.add(trf_name)
    if trf_hv_bus:
        keep.add(trf_hv_bus)
        keep.add(f"{trf_hv_bus}V_Source")   # HV‑side Vac source

    return keep

# ───────────────── main routine ───────────────────────────────────────────
def build_infinite_bus_islands(pf_data):
    """
    For every generator whose JSON flag `selected_for_tuning` is true:
      1. activate (or create) IntScheme «TUNE_<gen>»
      2. prune other generators (optional)
      3. wire that generator to an infinite bus (SMIB)
      4. deactivate – returns to the base network
    """
    try:
        snap = json.loads(_snap_path(pf_data).read_text())
    except Exception as e:
        raise RuntimeError(f"Snapshot JSON unreadable – {e}")

    todo = [g for g in snap["generators"] if g.get("selected_for_tuning")]
    if not todo:
        print("No generators marked → nothing to isolate."); return

    for meta in todo:
        gname  = meta["name"]
        var_id = f"TUNE_{gname}"

        # 1) create / fetch variant and activate it
        tune_var, var_expansion = _get_or_make_variant(pf_data, var_id)
        
        print(f"\n🔀 variant «{var_id}» activated")

        # ------------------------------------------------------------------
        # 2) keep only the objects we want in this tuning variant
        # ------------------------------------------------------------------
        keep = _keep_set(meta)

        for elm in pf_data.app.GetCalcRelevantObjects(
                '*.ElmSym,*.ElmGenstat,*.ElmTerm,*.ElmTr2,*.ElmVac,'
                '*.ElmLod,*.ElmLne'):
            if elm.loc_name not in keep:
                elm.Delete()

        print("   ↪ kept:", ", ".join(sorted(keep)))


        # 4) deactivate → drops back to the base network
        #var_expansion.Deactivate()
        tune_var.Deactivate()





