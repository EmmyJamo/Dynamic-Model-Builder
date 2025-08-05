# PowerFactory_Interaction/Tune_Isolated_Gens.py
# ────────────────────────────────────────────────────────────────────────────
# Seed AVR ElmDsl parameters for ONE generator variant.
# Called by the wrapper via  _seed_avr_parameters(...)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Dict, Any, List, Iterable, Optional

# ────────────────────────────────────────────────────────────────────────────
# 1) Canonical PARAM  →  attribute‑tag fallback chain
#    (first tag that works wins; extend if your template uses others)
# ────────────────────────────────────────────────────────────────────────────
_ATTR_MAP: List[tuple[str, tuple[str, ...]]] = [
    ("Tr",    ("par_Tr",    "Tr",    "e:Tr")),
    ("Ka",    ("par_Ka",    "Ka",    "e:Ka")),
    ("Ta",    ("par_Ta",    "Ta",    "e:Ta")),
    ("Ke",    ("par_Ke",    "Ke",    "e:Ke")),
    ("Te",    ("par_Te",    "Te",    "e:Te")),
    ("Kf",    ("par_Kf",    "Kf",    "e:Kf")),
    ("Tf",    ("par_Tf",    "Tf",    "e:Tf")),
    ("E1",    ("par_E1",    "E1")),
    ("Se1",   ("par_Se1",   "Se1")),
    ("E2",    ("par_E2",    "E2")),
    ("Se2",   ("par_Se2",   "Se2")),
    ("Vrmax", ("par_Vrmax", "Vrmax", "e:Vrmax")),
    ("Vrmin", ("par_Vrmin", "Vrmin", "e:Vrmin")),
]
_PARAM_NAMES       = [p for p, _ in _ATTR_MAP]
_ATTR_FALLBACKS    = {p: tags for p, tags in _ATTR_MAP}

# ────────────────────────────────────────────────────────────────────────────
# Lightweight PF helpers
# ────────────────────────────────────────────────────────────────────────────
def _try_set(obj, tag: str, val) -> bool:
    """Return True if SetAttribute succeeds (error code 0)."""
    try:
        return obj.SetAttribute(tag, val) == 0
    except Exception:
        return False

def _try_get(obj, tag: str):
    """Return attribute value or None if unavailable."""
    try:
        return obj.GetAttribute(tag)
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────
# Locate AVR ElmDsl block
# ────────────────────────────────────────────────────────────────────────────
def _locate_avr_block(pf_data, meta: Dict[str, Any]):
    gname    = meta["name"]
    avr_hint = (meta.get("AVR_Name") or "").lower()

    blocks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")  # type: ignore

    # 1) exact recorded name
    if avr_hint:
        for b in blocks:
            if b.loc_name.lower() == avr_hint:
                return b

    # 2) contains 'avr' + generator name
    for b in blocks:
        ln = b.loc_name.lower()
        if "avr" in ln and gname.lower() in ln:
            return b

    # 3) single AVR block in whole project?
    avr_blks = [b for b in blocks if "avr" in b.loc_name.lower()]
    if len(avr_blks) == 1:
        return avr_blks[0]

    if avr_blks:
        print(f"   ⚠️ multiple AVR blocks; using first: {avr_blks[0].loc_name}")
        return avr_blks[0]

    raise RuntimeError(f"AVR block not found for generator {gname}")

# ────────────────────────────────────────────────────────────────────────────
# Optional: write via the `params` vector if present & wanted
# (kept here in case you later prefer the list‑index approach)
# ────────────────────────────────────────────────────────────────────────────
def _maybe_write_via_params(avr_obj, p_dict: Dict[str, float]) -> bool:
    """
    Detect a writable `.params` attribute (list-like) and try to update there.
    Returns True if we actually wrote values this way; False otherwise.
    """
    if not hasattr(avr_obj, "params"):
        return False
    try:
        vec = list(avr_obj.params)  # copy
    except Exception:
        return False

    # Map JSON names -> index manually (taken from the UI order)
    index_map = {
        "Tr": 0, "Ka": 1, "Ta": 2, "Ke": 3, "Te": 4,
        "Kf": 5, "Tf": 6, "E1": 7, "Se1": 8, "E2": 9, "Se2": 10,
        "Vrmin": 11, "Vrmax": 12,
    }
    wrote_any = False
    for p, idx in index_map.items():
        if p in p_dict and idx < len(vec):
            vec[idx] = float(p_dict[p])
            wrote_any = True
    if wrote_any:
        avr_obj.params = vec  # type: ignore
    return wrote_any

# ────────────────────────────────────────────────────────────────────────────
# PUBLIC – seed parameters
# ────────────────────────────────────────────────────────────────────────────
def _seed_avr_parameters(pf_data,
                         meta: Dict[str, Any],
                         gname: str,
                         *,
                         json_key: str = "AVR_Seed",
                         dry_run: bool = False) -> bool:
    """
    Write parameters from JSON into the generator’s AVR ElmDsl.
    """
    print(f"      ↪ seeding AVR for «{gname}»")

    p_dict = (meta.get(json_key)
              or meta.get("AVR_Seed")
              or meta.get("AVR_Final"))
    if not p_dict:
        print(f"      ⚠️ no '{json_key}' (or fallback) params found – skip")
        return False

    try:
        avr_obj = _locate_avr_block(pf_data, meta)
    except Exception as e:
        print(f"      ⚠️ AVR locate failed: {e}")
        return False

    # quick peek
    peek = {p: _try_get(avr_obj, _ATTR_FALLBACKS[p][0]) for p in _PARAM_NAMES}
    print(f"      current(first‑tag) = {peek}")

    if dry_run:
        print(f"      [dry] WOULD WRITE: {p_dict}")
        return True

    # 1) try the simple .params vector route first (fast & tidy)
    if _maybe_write_via_params(avr_obj, p_dict):
        print("      ✓ seeded via 'params' vector")
        return True

    # 2) fallback: attribute‑by‑attribute
    all_ok = True
    for p in _PARAM_NAMES:
        if p not in p_dict:
            continue
        val = float(p_dict[p])
        for tag in _ATTR_FALLBACKS[p]:
            if _try_set(avr_obj, tag, val):
                break
        else:
            print(f"         ⚠️ write fail {avr_obj.loc_name}.{p}")
            all_ok = False

    msg = "✓ seeded" if all_ok else "⚠️ seeded with some errors"
    print(f"      {msg} AVR params on {avr_obj.loc_name}")
    return all_ok
