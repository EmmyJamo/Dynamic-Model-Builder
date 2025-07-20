"""
pf_gather_extended.py

Augments Gather_Gens with additional synchronous-generator + AVR data
needed for analytic seeding of excitation controller parameters.

Safe to re-run: merges new data into existing snapshot JSON (if present).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------
_SNAPSHOT_DIR = Path(
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)


# ------------------------------------------------------------------
# low-level safe attribute getter
# ------------------------------------------------------------------
def getattr_pf(obj, *tags, default=None):
    """
    Try a sequence of attribute name strings (PF tag syntax) on *obj*.
    Return the first non-None value; else *default*.

    Example:
        xd = getattr_pf(typ, 'e:xd', 'e:Xd', default=None)
    """
    for tg in tags:
        try:
            val = obj.GetAttribute(tg)
        except Exception:
            continue
        if val not in (None, ""):
            return val
    return default


# ------------------------------------------------------------------
# detect AVR block & harvest its parameters (if exposed)
# ------------------------------------------------------------------
_AVR_PARAM_TAGS = [
    # name in JSON  : tuple of PF attribute tags to try in order
    ("Tr",    ("Tr", "e:Tr", "par_Tr")),
    ("Ka",    ("Ka", "e:Ka", "par_Ka")),
    ("Ta",    ("Ta", "e:Ta", "par_Ta")),
    ("Ke",    ("Ke", "e:Ke", "par_Ke")),
    ("Te",    ("Te", "e:Te", "par_Te")),
    ("Kf",    ("Kf", "e:Kf", "par_Kf")),
    ("Tf",    ("Tf", "e:Tf", "par_Tf")),
    ("E1",    ("E1", "par_E1")),
    ("Se1",   ("Se1", "par_Se1")),
    ("E2",    ("E2", "par_E2")),
    ("Se2",   ("Se2", "par_Se2")),
    ("Vrmax", ("Vrmax", "par_Vrmax", "Vmax")),
    ("Vrmin", ("Vrmin", "par_Vrmin", "Vmin")),
]

def _harvest_avr_params(avr_blk) -> Dict[str, Optional[float]]:
    parms: Dict[str, Optional[float]] = {}
    for pname, tglist in _AVR_PARAM_TAGS:
        val = getattr_pf(avr_blk, *tglist, default=None)
        try:
            val = float(val) if val is not None else None
        except Exception:
            val = None
        parms[pname] = val
    return parms


def _find_avr_block(comp_obj):
    """
    Given a synchronous generator *compound* object (e:c_pmod) return
    the 1st ElmDsl whose name includes 'avr' (case-insensitive).

    Returns (avr_block_or_None, avr_name_or_None)
    """
    if comp_obj is None:
        return None, None
    for blk in comp_obj.GetContents('*.ElmDsl'):
        if 'avr' in blk.loc_name.lower():
            return blk, blk.loc_name
    return None, None


# ------------------------------------------------------------------
# transformer connectivity helper
# ------------------------------------------------------------------
def _find_transformer(pf_data, gen_bus_name: str):
    """
    Does a 2-winding transformer exist whose LV terminal sits on gen_bus_name?
    Returns (has_trf:bool, trf_name:str|None, grid_bus_name:str|None)
    """
    for trf in pf_data.grid_folder.GetContents("*.ElmTr2"):
        try:
            term_lv = trf.GetAttribute("buslv")     # ElmTerm
            bus_lv  = term_lv.GetAttribute("cterm") # IntBus
            if bus_lv and bus_lv.loc_name == gen_bus_name:
                term_hv = trf.GetAttribute("bushv")
                bus_hv  = term_hv.GetAttribute("cterm")
                return True, trf.loc_name, (bus_hv.loc_name if bus_hv else None)
        except Exception:
            continue
    return False, None, None


# ------------------------------------------------------------------
# synchronous electrical data harvest
# ------------------------------------------------------------------
def _harvest_sym_electrical(sym):
    typ = sym.GetAttribute("typ_id")  # machine type object
    data = {
        "Xd"      : getattr_pf(typ, "e:xd", "e:Xd"),
        "Xd_prime": getattr_pf(typ, "e:xdtr", "e:xd1", "e:Xd1"),
        "Xd_dbl"  : getattr_pf(typ, "e:xd2", "e:xdss", "e:Xd2"),
        "Xq"      : getattr_pf(typ, "e:xq", "e:Xq"),
        "Xq_prime": getattr_pf(typ, "e:xqtr", "e:xq1", "e:Xq1"),
        "Xq_dbl"  : getattr_pf(typ, "e:xq2", "e:xqss", "e:Xq2"),
        "Td0p"    : getattr_pf(typ, "e:Td0p", "e:tdo1", "e:Td0d"),
        "Td0pp"   : getattr_pf(typ, "e:Td0pp", "e:tdo2"),
        "Tq0p"    : getattr_pf(typ, "e:Tq0p", "e:tqo1"),
        "Tq0pp"   : getattr_pf(typ, "e:Tq0pp", "e:tqo2"),
        "H"       : getattr_pf(typ, "e:H", "e:tn"),   # be careful: tn≠H but we record raw
        "D"       : getattr_pf(typ, "e:dampr", "e:D"),
    }
    # cast floats where possible
    for k,v in data.items():
        try:
            data[k] = float(v)
        except Exception:
            data[k] = None
    return data


# ------------------------------------------------------------------
# operating-point signals (non-intrusive snapshot)
# ------------------------------------------------------------------
def _harvest_sym_operating(sym):
    """
    Some signals are measured at the generator terminal or within
    the machine element after a loadflow has been solved.
    """
    op = {
        "P_MW"   : getattr_pf(sym, "m:pgini"),
        "Q_MVar" : getattr_pf(sym, "m:qgini"),
        "Ifd_pu" : getattr_pf(sym, "m:ifd"),      # may be A or pu; depends
        "Efd_pu" : getattr_pf(sym, "m:efd"),      # some models expose this
        "Vt_pu"  : getattr_pf(sym, "m:usetp"),
    }
    for k,v in op.items():
        try:
            op[k] = float(v)
        except Exception:
            op[k] = None
    return op


# ------------------------------------------------------------------
# main gather
# ------------------------------------------------------------------
def Gather_Gens(pf_data):
    """
    Build / update the snapshot JSON with extended synchronous
    generator and AVR data required for analytic seeding.
    """
    # ---------- load existing snapshot (if any) ----------
    snap_path = _SNAPSHOT_DIR / f"{pf_data.project_name}_gen_snapshot.json"
    if snap_path.exists():
        with open(snap_path, encoding="utf-8") as fp:
            snap = json.load(fp)
    else:
        snap = {}

    # ensure keys exist
    gens_out = []

    # ---------- synchronous machines ----------
    for sym in pf_data.app.GetCalcRelevantObjects("*.ElmSym"):
        try:
            term = sym.GetAttribute("bus1")          # ElmTerm (cubicle)
            bus  = term.GetAttribute("cterm")        # IntBus
            bus_name = bus.loc_name if bus else None
        except Exception:
            bus_name = None

        # plant compound (to find controller blocks)
        comp_obj = sym.GetAttribute("e:c_pmod")
        avr_blk, avr_name = _find_avr_block(comp_obj)
        avr_parms = _harvest_avr_params(avr_blk) if avr_blk else {}

        # transformer connectivity
        has_trf, trf_name, grid_bus = _find_transformer(pf_data, bus_name or "")

        # machine type data
        typ = sym.GetAttribute("typ_id")
        MVA_rated = getattr_pf(typ, "e:sgn")
        kV_rated  = getattr_pf(typ, "e:ugn")

        # reactive limits
        qmax = getattr_pf(sym, "e:cQ_max")
        qmin = getattr_pf(sym, "e:cQ_min")

        # electrical constants
        elec = _harvest_sym_electrical(sym)
        oppt = _harvest_sym_operating(sym)

        rec = {
            "name"      : sym.loc_name,
            "type"      : "synchronous",
            "subtype"   : getattr_pf(sym, "e:cCategory"),
            "bus"       : bus_name,
            "MVA_rated" : float(MVA_rated) if MVA_rated is not None else None,
            "kV_rated"  : float(kV_rated)  if kV_rated  is not None else None,
            "MVar_Max"  : float(qmax)      if qmax is not None else None,
            "MVar_Min"  : float(qmin)      if qmin is not None else None,
            "Cubicle"   : term.loc_name if term else None,
            "AVR"       : bool(avr_blk),
            "AVR_Name"  : avr_name,
            "Has_Trf"   : has_trf,
            "Trf_Name"  : trf_name,
            "Grid_Bus"  : grid_bus,
            # nested dicts
            "AVR_Params": avr_parms,
            "Machine_Const": elec,
            "Operating_Point": oppt,
        }
        gens_out.append(rec)
        print(f"✓ Gathered synchronous gen: {sym.loc_name}")

    # ---------- inverter machines (stub: unchanged from your original) ----------
    for vsc in pf_data.app.GetCalcRelevantObjects("*.ElmGenstat"):
        try:
            term = vsc.GetAttribute("bus1")
            bus  = term.GetAttribute("cterm")
            bus_name = bus.loc_name if bus else None
        except Exception:
            bus_name = None

        has_trf, trf_name, grid_bus = _find_transformer(pf_data, bus_name or "")

        typ = vsc.GetAttribute("typ_id")
        rec = {
            "name"      : vsc.loc_name,
            "type"      : "inverter",
            "subtype"   : getattr_pf(vsc, "e:cCategory"),
            "bus"       : bus_name,
            "MVA_rated" : getattr_pf(typ, "e:sgn"),
            "kV_rated"  : getattr_pf(typ, "e:ugn"),
            "MVar_Max"  : getattr_pf(vsc, "e:cQ_max"),
            "MVar_Min"  : getattr_pf(vsc, "e:cQ_min"),
            "Cubicle"   : term.loc_name if term else None,
            "Has_Trf"   : has_trf,
            "Trf_Name"  : trf_name,
            "Grid_Bus"  : grid_bus,
            # TODO: add inverter control params later
        }
        gens_out.append(rec)
        print(f"✓ Gathered inverter gen: {vsc.loc_name}")

    # ---------- merge with existing snapshot ----------
    snap["generators"] = gens_out if gens_out else snap.get("generators", [])
    snap["Gen_Extended_ts"] = datetime_now_iso()

    # write
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snap_path, "w", encoding="utf-8") as fp:
        json.dump(snap, fp, indent=2)
    print(f"\n📦 Extended generator snapshot written to:\n    {snap_path}")

    return snap


# ------------------------------------------------------------------
# util: timestamp
# ------------------------------------------------------------------
from datetime import datetime
def datetime_now_iso():
    return datetime.now().isoformat(timespec="seconds")
