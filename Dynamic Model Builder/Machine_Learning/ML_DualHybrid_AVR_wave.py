"""
ML_DualHybrid_AVR_wave.py
------------------------------------------------------------

Dual‑candidate (PSO + CMA) ask/tell helper for AVR tuning.

Wrapper usage (per generator):
    ML.prepare_hybrid(pf_data, meta)            # idempotent
    cand_pso, cand_cma = ML.ask_two(gname)      # two candidates
    #  … wrapper runs RMS + scores each …
    best_src, best_vec = ML.tell_dual(
        scenario, gname, score_pso, score_cma,
        cand_pso=cand_pso, cand_cma=cand_cma
    )
    ML.write_candidate(pf_data, gname, best_vec)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# import the heavy‑lifting core module (unchanged)
# ---------------------------------------------------------------------------
import Machine_Learning.ML_DualHybrid_AVR as _core

TUNED_TAGS = _core.TUNED_TAGS   # re‑export for convenience

# ---------------------------------------------------------------------------
# One‑time initialiser
# ---------------------------------------------------------------------------
def prepare_hybrid(pf_data, meta: Dict[str, Any]) -> List[str]:
    """
    Ensure a hybrid optimiser object exists for this generator.
    Does nothing if it’s already cached in _core._HYBRID_REG.
    """
    gname = meta["name"]
    _core._tune_avr_parameters(
        pf_data, meta,
        iteration=1, current_score=math.inf,
        dry_run=True          # just seeds internal state
    )
    return TUNED_TAGS

# ---------------------------------------------------------------------------
# Produce two fresh candidates – one from each engine
# ---------------------------------------------------------------------------
def ask_two(gname: str) -> Tuple[List[float], List[float]]:
    """
    Returns `(vec_pso, vec_cma)` without writing them to PF.
    """
    hst = _core._HYBRID_REG.get(gname)
    if hst is None:
        raise RuntimeError(f"Hybrid state not initialised for {gname}. Call prepare_hybrid first.")

    pso_vec = _core._ask_from_pso(hst.pso)
    cma_vec = _core._ask_from_cma(hst.cma)

    # record – helps the core map scores back to the right particles
    hst.pso._pending = [pso_vec]
    hst.cma._pending = [cma_vec]

    return pso_vec, cma_vec

# ---------------------------------------------------------------------------
# Feed back the two scores and learn
# ---------------------------------------------------------------------------
def tell_dual(
    scenario: str,
    gname: str,
    score_pso: float,
    score_cma: float,
    *,
    cand_pso: Optional[List[float]] = None,
    cand_cma: Optional[List[float]] = None,
) -> Tuple[str, List[float]]:
    """
    Supply the two fitness values obtained by the wrapper, update the
    internal PSO/CMA states and return

        (best_source, best_vector)

    where *best_source* is "PSO" or "CMA".
    """
    hst = _core._HYBRID_REG.get(gname)
    if hst is None:
        raise RuntimeError(f"Hybrid state not initialised for {gname}")

    # keep last scenario on record – handy if you later want per‑scenario stats
    hst._last_scenario = scenario            # ← NEW (non‑intrusive)

    # overwrite stored candidates if wrapper passed modified vectors
    if cand_pso is not None:
        hst.pso._pending = [cand_pso]
    if cand_cma is not None:
        hst.cma._pending = [cand_cma]

    # delegate score handling to core helpers
    _core._feedback_score(gname, "PSO", score_pso)
    _core._feedback_score(gname, "CMA", score_cma)

    # credit / cross‑seed bookkeeping (unchanged helpers in core)
    _core._update_credit(hst, "PSO", score_pso)
    _core._update_credit(hst, "CMA", score_cma)
    _core._maybe_cross_seed(hst)

    # who is better right now?
    best_vec = _core.get_best_so_far(gname)
    best_src = "CMA" if hst.cma.best_f < hst.pso.best_f else "PSO"
    return best_src, best_vec

# ---------------------------------------------------------------------------
# Tiny convenience helpers
# ---------------------------------------------------------------------------
def vector_to_param_dict(vec: List[float]) -> Dict[str, float]:
    """List‑to‑dict mapping in the canonical tag order."""
    return {tag: float(vec[i]) for i, tag in enumerate(TUNED_TAGS)}

def write_candidate(pf_data, gname: str, vec: List[float]) -> bool:
    """Push an arbitrary vector into the AVR ElmDsl block."""
    return _core._write_params_to_avr(pf_data, gname, vector_to_param_dict(vec))
