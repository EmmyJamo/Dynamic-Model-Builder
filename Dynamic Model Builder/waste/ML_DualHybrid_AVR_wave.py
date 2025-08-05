"""
ML_DualHybrid_AVR_wave.py
─────────────────────────────────────────────────────────────────────────────
Dual‑candidate (PSO + CMA) helper that sits **on top** of
Machine_Learning.ML_DualHybrid_AVR (“the core”).

It exposes an ask / tell API so the outer wrapper can

  • obtain two fresh candidates (one PSO, one CMA),
  • run its own RMS simulations & scoring,
  • feed the scores back so the optimisers learn.

Nothing in here touches PowerFactory objects directly.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# import the heavy‑lifting core module
# ---------------------------------------------------------------------------
import Machine_Learning.ML_DualHybrid_AVR as _core

# Re‑export the canonical parameter tag order so callers have it handy
TUNED_TAGS = _core.TUNED_TAGS

# ---------------------------------------------------------------------------
# 1) one‑time initialiser – make sure a HybridController exists
# ---------------------------------------------------------------------------
def prepare_hybrid(pf_data, meta: Dict[str, Any]) -> List[str]:
    """
    Idempotent.  Creates and caches a HybridController for the generator
    described by *meta* (key = meta["name"]) – or returns silently if it
    already exists.

    Returns the list of tuned parameter tags (handy for callers).
    """
    _core.ensure_hybrid_state(pf_data, meta)     # new public helper in core
    return TUNED_TAGS

# ---------------------------------------------------------------------------
# 2) produce exactly two fresh candidate vectors
# ---------------------------------------------------------------------------
def ask_two(gname: str) -> Tuple[List[float], List[float]]:
    """
    Returns (vec_pso, vec_cma) **without** writing them to PowerFactory.

    Raises RuntimeError if `prepare_hybrid()` was not called first.
    """
    hst = _core._HYBRID_STATES.get(gname)
    if hst is None:
        raise RuntimeError(f"Hybrid optimiser not initialised for {gname}")

    vec_pso = _core._ask_from_pso(hst.pso)
    vec_cma = _core._ask_from_cma(hst.cma)

    # remember which vectors we asked for – helps score mapping
    hst.pso._pending = [vec_pso]
    hst.cma._pending = [vec_cma]

    return vec_pso, vec_cma

# ---------------------------------------------------------------------------
# 3) feed back the two scores and advance the optimiser
# ---------------------------------------------------------------------------
def tell_dual(
    scenario: str,                         # fast_dip | slow_hold  (kept for stats)
    gname: str,
    score_pso: float,
    score_cma: float,
    *,                                      # keyword‑only extras
    cand_pso: Optional[List[float]] = None,
    cand_cma: Optional[List[float]] = None,
) -> Tuple[str, List[float]]:
    """
    Update internal PSO & CMA states with the two fitness values and return:

        (best_source, best_vector)

    where *best_source* is "PSO" or "CMA".
    """
    hst = _core._HYBRID_STATES.get(gname)
    if hst is None:
        raise RuntimeError(f"Hybrid optimiser not initialised for {gname}")

    # record the scenario – handy if you ever want to inspect per‑scenario stats
    hst._last_scenario = scenario

    # if caller supplied explicit vectors (e.g. modified on write) – use them
    if cand_pso is not None:
        hst.pso._pending = [cand_pso]
    if cand_cma is not None:
        hst.cma._pending = [cand_cma]

    # delegate “score → optimiser” bookkeeping to core helpers
    _core._feedback_score(gname, "PSO", score_pso)
    _core._feedback_score(gname, "CMA", score_cma)

    _core._update_credit(hst, "PSO", score_pso)
    _core._update_credit(hst, "CMA", score_cma)
    _core._maybe_cross_seed(hst)

    # global best so far
    best_vec = _core.get_best_so_far(gname)
    best_src = "CMA" if hst.cma.best_f < hst.pso.best_f else "PSO"
    return best_src, best_vec

# ---------------------------------------------------------------------------
# 4) tiny conveniences for the outer wrapper
# ---------------------------------------------------------------------------
def vector_to_param_dict(vec: List[float]) -> Dict[str, float]:
    """Convert a vector (in TUNED_TAGS order) to a {tag: value} dict."""
    return {tag: float(vec[i]) for i, tag in enumerate(TUNED_TAGS)}

def write_candidate(pf_data, gname: str, vec: List[float]) -> bool:
    """
    Directly push *vec* into the AVR block of *gname* via the core helper.
    Returns True on success.
    """
    return _core._write_params_to_avr(
        pf_data, gname, vector_to_param_dict(vec)
    )
