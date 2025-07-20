# in PowerFactory_Interaction/Tune_Isolated_Gens.py
# (or whatever module you import as TUNE in the wrapper)

from __future__ import annotations
from typing import Dict, Any, Iterable

# ---------------------------------------------------------------------------
# AVR param names we might seed (include only what you store in JSON)
# ---------------------------------------------------------------------------
_AVR_PARAM_TAGS: Iterable[str] = (
    "Tr", "Ka", "Ta", "Vrmax", "Vrmin", "Ke", "Te",
    "Kf", "Tf", "E1", "E2", "Se1", "Se2"
)

# Fallback attribute tag patterns to try (in order) when writing each param.
# Extend if your PF model uses e.g. 'par_<name>' or 'c:<name>'.
_AVR_ATTR_FALLBACKS = {
    p: (f"e:{p}", p, f"c:{p}") for p in _AVR_PARAM_TAGS
}


# ---------------------------------------------------------------------------
# Lightweight PF helpers (safe/no‑crash versions)
# ---------------------------------------------------------------------------
def _try_set(obj, tag: str, val) -> bool:
    try:
        return obj.SetAttribute(tag, val) == 0
    except Exception:
        return False

def _try_get(obj, tag: str):
    try:
        return obj.GetAttribute(tag)
    except Exception:
        return None


def _locate_avr_block(pf_data, meta: Dict[str, Any]):
    """
    Locate the AVR ElmDsl object for the generator described by *meta*.

    Priority:
      1. meta["AVR_Name"] exact (case‑insensitive) match across *.ElmDsl.
      2. Any ElmDsl containing both 'avr' and the generator name.
      3. If exactly one ElmDsl contains 'avr', take it.
      4. If several 'avr' blocks exist, take the first but warn.

    Raises RuntimeError if nothing plausible is found.
    """
    gname    = meta["name"]
    avr_hint = (meta.get("AVR_Name") or "").lower()

    try:
        blocks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
    except Exception as e:
        raise RuntimeError(f"AVR search failed (GetCalcRelevantObjects): {e}")

    # 1) exact recorded name
    if avr_hint:
        for blk in blocks:
            if blk.loc_name.lower() == avr_hint:
                return blk

    # 2) contains 'avr' + gen name
    for blk in blocks:
        ln = blk.loc_name.lower()
        if "avr" in ln and gname.lower() in ln:
            return blk

    # 3) single 'avr' block in whole project?
    avr_blks = [b for b in blocks if "avr" in b.loc_name.lower()]
    if len(avr_blks) == 1:
        return avr_blks[0]

    # 4) ambiguous multi‑match
    if avr_blks:
        print(f"   ⚠️ multiple AVR blocks; using first: {avr_blks[0].loc_name}")
        return avr_blks[0]

    raise RuntimeError(f"AVR block not found for generator {gname}")


# ---------------------------------------------------------------------------
# PUBLIC: seed a *single* generator’s AVR (wrapper calls this)
# ---------------------------------------------------------------------------
def _seed_avr_parameters(pf_data,
                         meta: Dict[str, Any],
                         gname: str,
                         *,
                         json_key: str = "AVR_Seed",
                         dry_run: bool = False) -> bool:
    """
    Seed the AVR parameters for ONE generator.

    This is the function your wrapper calls:
        TUNE._seed_avr_parameters(pf_data, meta, gname)

    Parameters
    ----------
    pf_data : your PF context wrapper
    meta    : generator metadata dict from snapshot JSON
    gname   : generator name (redundant w/ meta["name"], kept for wrapper API)
    json_key: which JSON dict to read ("AVR_Seed" default; fallback to "AVR_Final")
    dry_run : True ⇒ print what *would* be written; no PF changes

    Returns
    -------
    bool  True if all available params written; False if error(s).
    """
    print(f"      ↪ seeding AVR for «{gname}»")

    # Pull param set from JSON
    p_dict = meta.get(json_key) or meta.get("AVR_Seed") or meta.get("AVR_Final")
    if not p_dict:
        print(f"      ⚠️ no '{json_key}' (or fallback) params in snapshot – skip.")
        return False

    # Locate the AVR block
    try:
        avr_obj = _locate_avr_block(pf_data, meta)
    except Exception as e:
        print(f"      ⚠️ AVR locate failed: {e}")
        return False

    # Show current values (quick peek, first tag only) for debug
    cur_peek = {}
    for p in _AVR_PARAM_TAGS:
        tag0 = _AVR_ATTR_FALLBACKS[p][0]
        cur_peek[p] = _try_get(avr_obj, tag0)
    print(f"      current(first‑tag) = {cur_peek}")

    if dry_run:
        print(f"      [dry] WOULD WRITE: {p_dict}")
        return True

    # Write each parameter that exists in JSON
    all_ok = True
    for p in _AVR_PARAM_TAGS:
        if p not in p_dict:
            continue
        val = float(p_dict[p])
        wrote = False
        for tag in _AVR_ATTR_FALLBACKS[p]:
            if _try_set(avr_obj, tag, val):
                wrote = True
                break
        if not wrote:
            print(f"         ⚠️ write fail {avr_obj.loc_name}.{p}")
            all_ok = False

    if all_ok:
        print(f"      ✓ seeded AVR params on {avr_obj.loc_name}")
    else:
        print(f"      ⚠️ seeded AVR params on {avr_obj.loc_name} with some errors")

    return all_ok

