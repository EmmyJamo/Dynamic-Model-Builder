# PowerFactory_Interaction/write_avr_params.py
# --------------------------------------------

from __future__ import annotations
from typing import Dict, Any

# Tag → index mapping for your avr_IEEET1 template
_PARAM_INDEX = {
    "Tr"   : 0,
    "Ka"   : 1,
    "Ta"   : 2,
    "Ke"   : 3,
    "Te"   : 4,
    "Kf"   : 5,
    "Tf"   : 6,
    "E1"   : 7,
    "Se1"  : 8,
    "E2"   : 9,
    "Se2"  : 10,
    "Vrmin": 11,
    "Vrmax": 12,
}

def write_avr_params(pf_data,
                     avr_obj,                # *.ElmDsl object (already located)
                     param_dict: Dict[str, float],
                     verbose: bool = True) -> bool:
    """
    Overwrite selected positions in avr_obj.params.
    Only keys present in `param_dict` are touched.
    """
    try:
        cur = list(avr_obj.params)        # PF → Python list
    except Exception as e:
        print(f"⚠️  cannot read params list – {e}")
        return False

    ok = True
    for tag, val in param_dict.items():
        if tag not in _PARAM_INDEX:
            if verbose:
                print(f"   ⚠️ unknown tag '{tag}', skipped")
            ok = False
            continue
        idx = _PARAM_INDEX[tag]
        if idx >= len(cur):
            if verbose:
                print(f"   ⚠️ index {idx} out of range for '{tag}'")
            ok = False
            continue
        cur[idx] = float(val)

    try:
        avr_obj.params = cur              # assign back in one go
        if verbose:
            print(f"   ✓ wrote {len(param_dict)} AVR fields via params[]")
    except Exception as e:
        print(f"⚠️  write‑back failed – {e}")
        return False

    return ok

