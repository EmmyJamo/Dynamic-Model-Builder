
"""
Tune_Isolated_Gens.py

Generator-by-generator AVR seeding + ML-style iterative tuning for isolated
variants (SMIB-style) in DIgSILENT PowerFactory.

Features
--------
• Loads project snapshot JSON (generators + Thevenin + seeds).
• Activates variant TUNE_<Gen>.IntScheme (must be pre-built).
• Locates AVR block, voltage source, and parameter events.
• Seeds AVR params from snapshot (dry-run aware).
• Iteratively tunes AVR params (placeholder ML state; swap in PSO+CMA later).
• Runs RMS + scoring via external wrappers (caller-supplied).
• Full DRY-RUN traversal diagnostic (no writes, no sims) to validate plumbing.

Edit points marked ### TODO ###.
"""

from __future__ import annotations

import json
import math
import random
import time
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

# ---------------------------------------------------------------------------
# External dependencies expected in your environment
# ---------------------------------------------------------------------------
# NOTE: these imports assume your existing project package layout.
# Adjust relative imports if packaging changes.
import PowerFactory_Control.Run_RMS_Sim as R_RMS_Sim          # for quick_rms_run + get_bus_thevenin
import Data_Scoring.Voltage.V_P as EV                         # your scoring function
# The module we are in (self-import pattern) is fine; but if you split seeding
# & tuning into separate modules, adjust these imports.
# ---------------------------------------------------------------------------


# ===========================================================================
# CONFIG / PATH HELPERS
# ===========================================================================
_SNAP_BASE = (
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)

def _snap_path(pf_data) -> Path:
    """Return snapshot JSON path for the active PF project."""
    return Path(_SNAP_BASE) / f"{pf_data.project_name}_gen_snapshot.json"


# ===========================================================================
# AVR SEED DATA MODEL
# ===========================================================================
@dataclass
class AVRSeed:
    Ka:   float = 200.0
    Ta:   float = 0.02
    Tr:   float = 0.01
    Ke:   float = 1.0
    Te:   float = 0.5
    Kf:   float = 0.0
    Tf:   float = 1.0
    Vrmax: float = 5.0
    Vrmin: float = -5.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AVRSeed":
        if not d:
            return cls()
        # ignore unknown keys
        kwargs = {f: float(d.get(f, getattr(cls, f))) for f in cls.__dataclass_fields__}
        return cls(**kwargs)

    # vector helpers for optimisers
    def to_vector(self) -> List[float]:
        return [self.Ka, self.Ta, self.Tr, self.Ke, self.Te, self.Kf, self.Tf, self.Vrmax, self.Vrmin]

    @classmethod
    def from_vector(cls, v: List[float]) -> "AVRSeed":
        return cls(*v[:9])  # truncate/ignore extras


# Ordered list of the parameters we normally tune.
_AVR_PARAM_TAGS = ["Ka", "Ta", "Tr", "Ke", "Te", "Kf", "Tf", "Vrmax", "Vrmin"]

# ---------------------------------------------------------------------------
# Attribute name map: AVR parameter name -> PF attribute tag on ElmDsl
# ---------------------------------------------------------------------------
# ⚠️  UPDATE THESE TAGS to match your *actual* AVR model!
# Many PF IEEE exciters use e.g. 'c:Ka' or 'par_Ka' etc. Use dry-run to inspect.
# We'll try 3 patterns in _write_avr_params/_read_avr_params automatically.
AVR_ATTR_MAP = {
    "Ka":    ["e:Ka", "c:Ka", "Ka"],
    "Ta":    ["e:Ta", "c:Ta", "Ta"],
    "Tr":    ["e:Tr", "c:Tr", "Tr"],
    "Ke":    ["e:Ke", "c:Ke", "Ke"],
    "Te":    ["e:Te", "c:Te", "Te"],
    "Kf":    ["e:Kf", "c:Kf", "Kf"],
    "Tf":    ["e:Tf", "c:Tf", "Tf"],
    "Vrmax": ["e:Vrmax", "c:Vrmax", "Vrmax"],
    "Vrmin": ["e:Vrmin", "c:Vrmin", "Vrmin"],
}


# ===========================================================================
# SNAPSHOT LOAD / SAVE
# ===========================================================================
def _load_snapshot(pf_data) -> Dict[str, Any]:
    path = _snap_path(pf_data)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    try:
        with open(path, encoding="utf-8") as fp:
            snap = json.load(fp)
        return snap
    except Exception as e:
        raise RuntimeError(f"Could not read snapshot: {e}")

def _save_snapshot(pf_data, snap: Dict[str, Any]) -> None:
    path = _snap_path(pf_data)
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(snap, fp, indent=2)
        print(f"   📁 snapshot updated: {path}")
    except Exception as e:
        print(f"   ⚠️ snapshot write failed: {e}")


# ===========================================================================
# POWERFACTORY OBJECT LOCATION UTILITIES
# ===========================================================================
def _activate_variant(pf_data, gen_name: str):
    """Activate tuning variant TUNE_<GenName>.IntScheme."""
    var_id = f"TUNE_{gen_name}.IntScheme"
    try:
        scheme = pf_data.variations_folder.GetContents(var_id)[0]
    except Exception as e:
        raise RuntimeError(f"Variant {var_id} not found ({e})")
    try:
        scheme.Activate()
    except Exception as e:
        raise RuntimeError(f"Could not activate {var_id}: {e}")
    print(f"   ✓ variant «{scheme.loc_name}» activated")
    return scheme


def _locate_voltage_source(pf_data, bus: str):
    """Return ElmVac object named '<bus>V_Source.ElmVac' or None."""
    vs_name = f"{bus}V_Source.ElmVac"
    try:
        lst = pf_data.grid_folder.GetContents(vs_name)
        if not lst:
            print(f"   ⚠️ voltage source {vs_name} not found")
            return None
        return lst[0]
    except Exception as e:
        print(f"   ⚠️ error locating {vs_name}: {e}")
        return None


def _locate_param_event(pf_data, bus: str, kind: str):
    """
    kind = "Drop" | "Rise"
    Returns event object or None.
    """
    ev_name = f"Voltage {kind}{bus}.EvtParam"
    try:
        lst = pf_data.Simulation_Folder.GetContents(ev_name)
        if not lst:
            print(f"   ⚠️ param event missing: {ev_name}")
            return None
        return lst[0]
    except Exception as e:
        print(f"   ⚠️ error locating {ev_name}: {e}")
        return None


def _locate_avr_block(pf_data, meta: Dict[str, Any]):
    """
    Try to find AVR ElmDsl object for this generator.
    Strategy:
      1. If snapshot meta["AVR_Name"] present, try direct lookup in Plant composite.
      2. Else search all ElmDsl under the generator's Plant composite for 'avr' in name.
      3. Else brute search app.GetCalcRelevantObjects('*.ElmDsl') and match containing gen name.
    """
    gname = meta["name"]
    avr_hint = (meta.get("AVR_Name") or "").lower()

    # Try to get Plant composite from PF: we stored Plant in Gather_Gens? If not, fallback.
    # We'll scan everything under the active project; expensive but robust for now.
    try:
        all_blks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
    except Exception as e:
        raise RuntimeError(f"AVR search failed (all ElmDsl): {e}")

    # 1) exact name
    if avr_hint:
        for blk in all_blks:
            if blk.loc_name.lower() == avr_hint:
                return blk

    # 2) contains 'avr' + generator name near?
    for blk in all_blks:
        nm = blk.loc_name.lower()
        if "avr" in nm and gname.lower() in nm:
            return blk

    # 3) contains 'avr' only
    avr_candidates = [b for b in all_blks if "avr" in b.loc_name.lower()]
    if len(avr_candidates) == 1:
        return avr_candidates[0]

    # 4) bail
    raise RuntimeError(f"AVR block not found for generator {gname}")


# ===========================================================================
# AVR PARAM READ / WRITE WITH FALLBACK TAGS
# ===========================================================================
def _try_get(obj, tag: str):
    try:
        return obj.GetAttribute(tag)
    except Exception:
        return None

def _try_set(obj, tag: str, val) -> bool:
    try:
        return obj.SetAttribute(tag, val) == 0
    except Exception:
        return False

def _read_one_param(avr_obj, p: str):
    for tag in AVR_ATTR_MAP.get(p, []):
        v = _try_get(avr_obj, tag)
        if v is not None:
            return float(v)
    return None

def _read_avr_params(avr_obj) -> Dict[str, Optional[float]]:
    vals = {}
    for p in _AVR_PARAM_TAGS:
        vals[p] = _read_one_param(avr_obj, p)
    return vals

def _write_avr_params(avr_obj, params: Dict[str, float]) -> bool:
    all_ok = True
    for p, v in params.items():
        ok_one = False
        for tag in AVR_ATTR_MAP.get(p, []):
            if _try_set(avr_obj, tag, float(v)):
                ok_one = True
                break
        if not ok_one:
            print(f"      ⚠ write fail: {avr_obj.loc_name}.{p} (tried {AVR_ATTR_MAP.get(p)})")
            all_ok = False
    return all_ok


# ===========================================================================
# BOUNDS
# ===========================================================================
_DEFAULT_BOUNDS = {
    "Ka":    (10.0, 500.0),
    "Ta":    (0.001, 0.5),
    "Tr":    (0.001, 0.2),
    "Ke":    (0.1, 10.0),
    "Te":    (0.01, 5.0),
    "Kf":    (0.0, 5.0),
    "Tf":    (0.01, 5.0),
    "Vrmax": (1.0, 20.0),
    "Vrmin": (-20.0, -1.0),
}

def _get_bounds_for_gen(meta: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """
    Returns (lower_bounds, upper_bounds) arrays aligning with _AVR_PARAM_TAGS.
    Per-gen overrides:
        meta["AVR_Bounds"] = { "Ka": [min,max], ... }
    """
    lb, ub = [], []
    overrides = meta.get("AVR_Bounds", {}) or {}
    for p in _AVR_PARAM_TAGS:
        if p in overrides:
            lo, hi = overrides[p]
        else:
            lo, hi = _DEFAULT_BOUNDS[p]
        lb.append(float(lo))
        ub.append(float(hi))
    return lb, ub


# ===========================================================================
# ML OPTIMISER STATE (PLACEHOLDER)
# ===========================================================================
class _DummyHybridState:
    """
    Minimal stand-in for PSO + CMA to prove the plumbing.
    * ask() returns a random point within bounds (w/ shrink toward best).
    * tell(x,f) records best.
    Replace with real PSO/CMA engine when ready.
    """
    def __init__(self, seed_vec, lb, ub):
        self.lb = lb[:]
        self.ub = ub[:]
        self.dim = len(lb)
        self.best_x = seed_vec[:]
        self.best_f = math.inf
        self._last_candidate = None
        self.iter = 0

    def ask(self):
        self.iter += 1
        cand = []
        for i in range(self.dim):
            lo, hi = self.lb[i], self.ub[i]
            # shrink around best as iter grows
            shrink = min(0.8, self.iter * 0.05)
            mid = self.best_x[i]
            span = (hi - lo) * (1.0 - shrink)
            lo2 = max(lo, mid - span/2)
            hi2 = min(hi, mid + span/2)
            cand.append(random.uniform(lo2, hi2))
        return cand

    def tell(self, x, f):
        if f < self.best_f:
            self.best_f = f
            self.best_x = x[:]


# global cache
_OPT_STATE: Dict[str, _DummyHybridState] = {}

def _get_state_for_gen(gname: str, seed_vec: List[float], lb, ub) -> _DummyHybridState:
    st = _OPT_STATE.get(gname)
    if st is None:
        st = _DummyHybridState(seed_vec, lb, ub)
        _OPT_STATE[gname] = st
    return st


# ===========================================================================
# SEED PARAMS → AVR (DRY-RUN AWARE)
# ===========================================================================
def _seed_avr_parameters(pf_data,
                         meta: Dict[str, Any],
                         bus: str,
                         *,
                         dry_run: bool = False) -> None:
    """
    Write precomputed AVR seed values from snapshot meta into the active variant.
    """
    gname = meta["name"]
    seed_dict = meta.get("AVR_Seed")
    if not seed_dict:
        raise KeyError(f"No AVR_Seed entry for {gname} in snapshot.")

    seed = AVRSeed.from_dict(seed_dict)
    params = seed.as_dict()

    # locate AVR block
    try:
        avr_obj = _locate_avr_block(pf_data, meta)
    except Exception as e:
        raise RuntimeError(f"Cannot seed AVR for {gname}: {e}")

    if dry_run:
        cur = _read_avr_params(avr_obj)
        print(f"      [dry] AVR current -> {cur}")
        print(f"      [dry] WOULD WRITE seed -> {params}")
        return

    ok = _write_avr_params(avr_obj, params)
    if not ok:
        raise RuntimeError(f"Failed to write one or more AVR params for {gname}")
    print(f"      seeded AVR params on {avr_obj.loc_name}")


# ===========================================================================
# ML TUNING STEP (DRY-RUN AWARE)
# ===========================================================================
def _tune_avr_parameters(pf_data,
                         meta: Dict[str, Any],
                         iter_idx: int,
                         last_score: float,
                         *,
                         dry_run: bool = False) -> bool:
    """
    One ML iteration: propose next candidate parameter vector and write it.
    Returns:
        True  if new params written -> caller should run RMS again.
        False if tuning finished / aborted.
    """
    gname = meta["name"]

    try:
        avr_obj = _locate_avr_block(pf_data, meta)
    except Exception as e:
        print(f"      ⚠️ AVR locate fail: {e}")
        return False

    # bounds
    lb, ub = _get_bounds_for_gen(meta)

    # seed vector
    seed_dict = meta.get("AVR_Seed") or {}
    seed = AVRSeed.from_dict(seed_dict)
    seed_vec = seed.to_vector()

    # state
    state = _get_state_for_gen(gname, seed_vec, lb, ub)

    # update from last candidate
    prev = getattr(state, "_last_candidate", None)
    if prev is not None:
        state.tell(prev, last_score)

    cand = state.ask()
    state._last_candidate = cand

    params = {p: float(cand[i]) for i, p in enumerate(_AVR_PARAM_TAGS)}

    if dry_run:
        cur = _read_avr_params(avr_obj)
        print(f"      [dry] AVR current -> {cur}")
        print(f"      [dry] ML cand -> {params}")
        return False

    ok = _write_avr_params(avr_obj, params)
    if not ok:
        print(f"      ⚠ write error; stopping tuning for {gname}")
        return False

    print("      ML cand:",
          ", ".join(f"{k}={params[k]:.4g}" for k in ("Ka","Ta","Tr","Ke","Te")))

    # optional per-gen cap
    max_allowed = int(meta.get("AVR_MaxIter", 20))
    if iter_idx >= max_allowed:
        print(f"      ⏹ reached iteration cap ({max_allowed}) for {gname}")
        return False

    return True


# ===========================================================================
# PARAM EVENT ENABLER
# ===========================================================================
def _enable_param_events(pf_data, bus: str, *, dry_run: bool = False) -> None:
    drop = _locate_param_event(pf_data, bus, "Drop")
    rise = _locate_param_event(pf_data, bus, "Rise")

    for ev, label in ((drop,"Drop"), (rise,"Rise")):
        if ev is None:
            continue
        if dry_run:
            print(f"   [dry] WOULD enable event {label}: {ev.loc_name}")
            continue
        try:
            ev.SetAttribute("e:outserv", False)
            print(f"   ✓ event {label} in service ({ev.loc_name})")
        except Exception as e:
            print(f"   ⚠️ could not enable {label} ({e})")



# ===========================================================================
# HIGH-LEVEL TUNING LOOP
# ===========================================================================
def tune_selected_generators(pf_data,
                             target_score: float = 0.00085,
                             max_iter: int = 5,
                             *,
                             dry_run: bool = False):
    """
    Iterate over all generators flagged  selected_for_tuning:true  in snapshot.
    For each:
      • activate variant
      • enable events
      • seed AVR
      • loop: RMS run, score, ML adjust, until target or max_iter

    dry_run=True → no PF writes, no RMS runs, shows what would happen.
    """
    snap = _load_snapshot(pf_data)

    todo = [g for g in snap.get("generators", []) if g.get("selected_for_tuning")]
    if not todo:
        print("📭  Nothing to tune – no generator carries the flag.")
        return

    print(f"🔧  Starting tuning loop for {len(todo)} generator(s)…")

    for meta in todo:
        gname = meta["name"]

        # select bus (HV if transformer)
        if meta.get("Has_Trf"):
            bus = meta.get("Grid_Bus") or meta.get("trf_hv_bus")
            if not bus:
                print(f"⚠️  {gname}: Has_Trf=True but no Grid_Bus in snapshot – skipping.")
                continue
            print(f"⚙️   {gname} transformer‑coupled – using HV bus «{bus}»")
        else:
            bus = meta["bus"]
            print(f"⚙️   {gname} directly connected – bus «{bus}»")

        print("\n" + "═"*70)
        print(f"⚙️   Tuning «{gname}»")

        # variant
        try:
            _activate_variant(pf_data, gname)
        except Exception as e:
            print(f"   ⚠️ cannot activate variant for {gname}: {e}")
            continue

        # voltage source: update to Thevenin
        vs_obj = _locate_voltage_source(pf_data, bus)
        if vs_obj is None:
            print(f"   ⚠️ skipping {gname} (no voltage source)")
            continue

        # Thevenin
        try:
            R, X, V_kV, U0 = R_RMS_Sim.get_bus_thevenin(pf_data, bus)
        except Exception as e:
            print(f"   ⚠️ Thevenin lookup failed for {bus}: {e}")
            continue

        if dry_run:
            print(f"   [dry] WOULD set {vs_obj.loc_name}: R1={R:.4f} X1={X:.4f} vtarget={U0:.4f}")
        else:
            ok = (
                _try_set(vs_obj, "e:R1", float(R)) and
                _try_set(vs_obj, "e:X1", float(X)) and
                _try_set(vs_obj, "e:vtarget", float(U0))
            )
            if not ok:
                print(f"   ⚠️ could not set R/X/vtarget on {vs_obj.loc_name}")

        # events
        _enable_param_events(pf_data, bus, dry_run=dry_run)

        # seed AVR
        try:
            _seed_avr_parameters(pf_data, meta, bus, dry_run=dry_run)
        except Exception as e:
            print(f"   ⚠️ seed fail: {e}")
            continue

        # iterative tuning
        last_score = math.inf
        for k in range(1, max_iter+1):
            if dry_run:
                print(f"   [dry] iter {k}: skip RMS + scoring")
                # pretend we got a gradually improving score
                last_score = max(target_score * 2 / k, target_score*1.1)
            else:
                try:
                    last_score = _run_rms_and_score(pf_data, bus)
                except Exception as e:
                    print(f"   ⚠️ RMS/score error: {e}")
                    break

            if last_score <= target_score:
                print(f"   🎉 target reached in {k} iteration(s) (score={last_score:.6f})")
                break

            changed = _tune_avr_parameters(pf_data, meta, k, last_score, dry_run=dry_run)
            if not changed:
                print(f"   ⏹ tuning stopped (iter={k}, score={last_score:.6f})")
                break

        # store result
        meta["final_score"]    = float(last_score)
        meta["tuning_done_ts"] = datetime.datetime.now().isoformat(timespec="seconds")

    # persist snapshot
    if dry_run:
        print("   [dry] NOT writing snapshot (dry_run=True)")
    else:
        _save_snapshot(pf_data, snap)

    print("\n✅  Tuning loop complete.")


# ===========================================================================
# DRY-RUN TRAVERSAL / DEBUGGER
# ===========================================================================
def dry_run_check(pf_data) -> None:
    """
    Validate that everything needed for tuning exists, without writing or
    simulating. Prints a detailed report per generator.
    """
    try:
        snap = _load_snapshot(pf_data)
    except Exception as e:
        print(f"📄 snapshot load failed: {e}")
        return

    todo = [g for g in snap.get("generators", []) if g.get("selected_for_tuning")]
    if not todo:
        print("📭  Nothing to dry-run: no generators selected_for_tuning.")
        return

    print(f"🔎 Dry-run: checking {len(todo)} generator(s)…\n")

    for meta in todo:
        gname = meta["name"]
        if meta.get("Has_Trf"):
            bus = meta.get("Grid_Bus") or meta.get("trf_hv_bus")
            if not bus:
                print(f"── {gname}: Has_Trf but no HV bus – SKIP\n")
                continue
            print(f"── {gname} (Has_Trf) → bus «{bus}»")
        else:
            bus = meta["bus"]
            print(f"── {gname} → bus «{bus}»")

        # variant
        var_id = f"TUNE_{gname}.IntScheme"
        try:
            scheme = pf_data.variations_folder.GetContents(var_id)[0]
            scheme.Activate()
            print(f"   ✓ variant activated ({scheme.loc_name})")
        except Exception as e:
            print(f"   ⚠️ variant {var_id} not found/activate fail ({e})")
            continue

        # voltage source
        vs = _locate_voltage_source(pf_data, bus)
        if vs:
            r1 = _try_get(vs, "e:R1")
            x1 = _try_get(vs, "e:X1")
            vt = _try_get(vs, "e:vtarget")
            print(f"   ✓ voltage source {vs.loc_name}: R1={r1} X1={x1} vtarget={vt}")
        else:
            print(f"   ⚠️ no voltage source for {bus}")

        # events
        for kind in ("Drop", "Rise"):
            ev = _locate_param_event(pf_data, bus, kind)
            if ev:
                print(f"   ✓ event {ev.loc_name}: outserv={getattr(ev,'outserv',None)}")
            else:
                print(f"   ⚠️ missing Voltage {kind} event")

        # AVR
        try:
            avr = _locate_avr_block(pf_data, meta)
            cur = _read_avr_params(avr)
            print(f"   AVR {avr.loc_name} current: {cur}")
        except Exception as e:
            print(f"   ⚠️ AVR locate/read fail: {e}")
            continue

        # seeds + bounds
        sd = meta.get("AVR_Seed")
        if sd:
            print(f"   AVR seed: {sd}")
        else:
            print("   ⚠️ no AVR_Seed in snapshot")

        lb, ub = _get_bounds_for_gen(meta)
        bds = {p: (lb[i], ub[i]) for i, p in enumerate(_AVR_PARAM_TAGS)}
        print(f"   bounds: {bds}\n")

    print("✅ Dry-run check complete.\n")


