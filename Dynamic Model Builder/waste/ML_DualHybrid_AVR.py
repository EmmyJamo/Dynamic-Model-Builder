"""
ML_DualHybrid_AVR.py
─────────────────────────────────────────────────────────────────────────────
Core dual‑hybrid optimiser (PSO + CMA‑ES) for AVR parameter tuning.

Used by
  • gen_tuner.py                       – legacy single‑step path
  • ML_DualHybrid_AVR_wave.py          – two‑candidate wave path

External deps
  pip install cma              # real CMA‑ES
  # pyswarms optional; current PSO is an internal lightweight engine
"""

from __future__ import annotations
import math, random, datetime
from typing import Dict, Any, List, Tuple, Optional

# ────────────────────────────────────────────────────────────────────────────
# External CMA‑ES library
# ────────────────────────────────────────────────────────────────────────────
import cma                                       # type: ignore

# ────────────────────────────────────────────────────────────────────────────
# AVR parameters (order matters everywhere)
# ────────────────────────────────────────────────────────────────────────────
TUNED_TAGS   = ["Ka", "Ta", "Tr", "Ke", "Te", "Kf", "Tf", "Vrmax", "Vrmin"]
_EXTRA_TAGS  = ["E1", "E2", "Se1", "Se2"]        # carried through, not tuned

# ---------------------------------------------------------------------------
# Default parameter bounds (ASCII minus signs only!)
# ---------------------------------------------------------------------------
_BOUNDS_DEFAULT: dict[str, tuple[float, float]] = {
    "Ka":    (10.0, 500.0),
    "Ta":    (1e-3, 0.5),
    "Tr":    (1e-3, 0.2),
    "Ke":    (0.1, 10.0),
    "Te":    (1e-2, 5.0),
    "Kf":    (0.0, 5.0),
    "Tf":    (1e-2, 5.0),
    "Vrmax": (1.0, 20.0),
    "Vrmin": (-20.0, -1.0),
}

# ═══════════════════════════════════════════════════════════════════════════
#  Light‑weight PSO (ask / tell friendly)
# ═══════════════════════════════════════════════════════════════════════════
class _PSO:
    """
    Very small “global‑best” PSO implementation that exposes .ask() / .tell().
    ask() returns **one** vector per call so the wrapper can interleave with
    CMA‑ES evaluations easily.
    """

    def __init__(
        self,
        seed: List[float],
        lb:   List[float],
        ub:   List[float],
        n_part: int  = 10,
        w: float     = 0.6,
        c1: float    = 1.5,
        c2: float    = 1.5,
    ):
        self.dim  = len(seed)
        self.lb   = lb
        self.ub   = ub
        self.w, self.c1, self.c2 = w, c1, c2

        self.pos: List[List[float]] = [seed[:]] + [
            [random.uniform(lb[i], ub[i]) for i in range(self.dim)]
            for _ in range(n_part - 1)
        ]
        self.vel: List[List[float]] = [[0.0] * self.dim for _ in range(n_part)]

        self.pbest = self.pos[:]
        self.pbest_f = [float("inf")] * n_part
        self.gbest   = seed[:]
        self.gbest_f = float("inf")

        self._queue: List[List[float]] = []     # ← initialise ask‑queue
        self._pending: List[List[float]] = []   # used by wave layer

    # ------------------------------------------------------------------ ask
    def ask(self) -> List[List[float]]:
        """
        Returns a list with **one** position vector.
        Iterates through the swarm in order; when the generation is exhausted
        a new one is created after velocity update.
        """
        if not self._queue:
            self._queue = [p[:] for p in self.pos]   # new generation snapshot
        return [self._queue.pop(0)]

    # ----------------------------------------------------------------- tell
    def tell(self, xs: List[List[float]], fs: List[float]):
        """
        xs must come back *in the same order* they were asked, one by one.
        Once the whole generation has been evaluated, velocities / positions
        are updated automatically for the next ask() cycle.
        """
        for x, f in zip(xs, fs):
            # locate particle by identity (safe because we copy positions)
            try:
                idx = next(i for i, p in enumerate(self.pos) if p is x or p == x)
            except StopIteration:
                # fall back to first unevaluated slot
                idx = self.pbest_f.index(math.inf)

            self.pos[idx] = x[:]

            if f < self.pbest_f[idx]:
                self.pbest_f[idx] = f
                self.pbest[idx]   = x[:]

            if f < self.gbest_f:
                self.gbest_f, self.gbest = f, x[:]

        # when queue empty → generation finished → update all particles
        if not self._queue:
            for i in range(len(self.pos)):
                for d in range(self.dim):
                    r1, r2 = random.random(), random.random()
                    cognitive = self.c1 * r1 * (self.pbest[i][d] - self.pos[i][d])
                    social    = self.c2 * r2 * (self.gbest[d]     - self.pos[i][d])
                    self.vel[i][d] = self.w * self.vel[i][d] + cognitive + social
                    self.pos[i][d] += self.vel[i][d]

                    # clamp to bounds
                    if self.pos[i][d] < self.lb[d]:
                        self.pos[i][d] = self.lb[d]
                        self.vel[i][d] = 0.0
                    elif self.pos[i][d] > self.ub[d]:
                        self.pos[i][d] = self.ub[d]
                        self.vel[i][d] = 0.0

# ═══════════════════════════════════════════════════════════════════════════
#  CMA‑ES wrapper  (real cma library, ask / tell)
# ═══════════════════════════════════════════════════════════════════════════
class _CMA:
    """
    Thin wrapper around `cma.CMAEvolutionStrategy` exposing ask()/tell()
    with **one** vector per ask, mirroring the PSO interface.
    """

    def __init__(
        self,
        seed: List[float],
        lb:   List[float],
        ub:   List[float],
        sigma_frac: float = 0.25,
        popsize:     int  = 6,
    ):
        self.lb, self.ub = lb, ub
        # initial sigma = average range * fraction
        sigma0 = sigma_frac * sum(ub[i] - lb[i] for i in range(len(lb))) / len(lb)

        opts = {
            "bounds":   [lb, ub],
            "popsize":  popsize,
            "verb_log": 0,
            "verb_disp": 0,
        }
        self.es = cma.CMAEvolutionStrategy(seed, sigma0, opts)
        self._asked: List[List[float]] = []          # queue
        self._pending: List[List[float]] = []        # used by wave layer

    def ask(self) -> List[List[float]]:
        if not self._asked:
            self._asked = self.es.ask()
        return [self._asked.pop(0)]

    def tell(self, xs: List[List[float]], fs: List[float]):
        # accumulate until full λ then pass back to CMA
        if not hasattr(self, "_acc_x"):
            self._acc_x, self._acc_f = [], []
        self._acc_x += xs
        self._acc_f += fs

        if len(self._acc_x) >= self.es.popsize:
            self.es.tell(self._acc_x, self._acc_f)
            self._acc_x, self._acc_f = [], []

    # convenient properties
    @property
    def best_f(self) -> float:
        return self.es.best.f

    @property
    def best_x(self) -> List[float]:
        return list(self.es.best.x)

# ═══════════════════════════════════════════════════════════════════════════
#  Hybrid container (coordinates PSO + CMA)
# ═══════════════════════════════════════════════════════════════════════════
class _Hybrid:
    def __init__(self, seed: List[float], lb: List[float], ub: List[float]):
        self.pso = _PSO(seed, lb, ub)
        self.cma = _CMA(seed, lb, ub)

        self.best_x: List[float] = seed[:]
        self.best_f = float("inf")

        self.credit_cma = 0.5                # % of iterations given to CMA
        self._last_scenario = "fast_dip"
        self._iter = 0

    # --------------------------------------------------------- book‑keeping
    def _update_global(self, src: str, vec: List[float], f: float):
        if f < self.best_f:
            self.best_f, self.best_x = f, vec[:]
            # simple credit adjustment
            if src == "CMA":
                self.credit_cma = min(0.9, self.credit_cma + 0.05)
            else:
                self.credit_cma = max(0.2, self.credit_cma - 0.05)

    # ------------------------------------------------------------- helpers
    def maybe_cross_seed(self, period: int = 5):
        """Push current global best into both engines periodically."""
        self._iter += 1
        if self._iter % period == 0:
            self.pso.gbest = self.best_x[:]
            self.cma.es.set_xmean(self.best_x[:])

# ═══════════════════════════════════════════════════════════════════════════
#  Global registry – one Hybrid per generator
# ═══════════════════════════════════════════════════════════════════════════
_HYBRID_STATES: Dict[str, _Hybrid] = {}

# ---------------------------------------------------------------------------
# Helper: bounds & seed extraction from snapshot meta
# ---------------------------------------------------------------------------
def _bounds_for(meta: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    lb, ub = [], []
    user_bounds = meta.get("AVR_Bounds", {})
    for tag in TUNED_TAGS:
        lo, hi = user_bounds.get(tag, _BOUNDS_DEFAULT[tag])
        lb.append(float(lo))
        ub.append(float(hi))
    return lb, ub


def _seed_vec(meta: Dict[str, Any], lb: List[float], ub: List[float]) -> List[float]:
    src = meta.get("AVR_Seed") or meta.get("AVR_Final") or {}
    return [float(src.get(tag, 0.5 * (lo + hi))) for tag, lo, hi in zip(TUNED_TAGS, lb, ub)]


def ensure_hybrid_state(pf_data, meta: Dict[str, Any]) -> _Hybrid:
    """
    Public helper: guarantees that `_HYBRID_STATES[gen]` exists and returns it.
    Safe to call multiple times (idempotent).
    """
    gname = meta["name"]
    if gname not in _HYBRID_STATES:
        lb, ub = _bounds_for(meta)
        seed   = _seed_vec(meta, lb, ub)
        _HYBRID_STATES[gname] = _Hybrid(seed, lb, ub)
    return _HYBRID_STATES[gname]

# ---------------------------------------------------------------------------
#  (tiny) helpers used by the “wave” adapter
# ---------------------------------------------------------------------------
def _ask_from_pso(pso: _PSO) -> List[float]:
    return pso.ask()[0]

def _ask_from_cma(cma: _CMA) -> List[float]:
    return cma.ask()[0]

def _feedback_score(gname: str, src: str, score: float):
    """
    Feed back a single score to the proper engine.  Assumes the most recent
    candidate for that engine is stored in `_pending` (set by ask_two()).
    """
    h = _HYBRID_STATES[gname]

    if src == "PSO":
        vec = h.pso._pending.pop(0)
        h.pso.tell([vec], [score])
    else:
        vec = h.cma._pending.pop(0)
        h.cma.tell([vec], [score])

    h._update_global(src, vec, score)

def _update_credit(h: _Hybrid, src: str, score: float):
    h._update_global(src, h.best_x, score)    # just reuse the same logic

def _maybe_cross_seed(h: _Hybrid):
    h.maybe_cross_seed()

def get_best_so_far(gname: str) -> List[float]:
    return _HYBRID_STATES[gname].best_x[:]

# ---------------------------------------------------------------------------
#  Write parameters into PF (shared by wrapper & wave)
# ---------------------------------------------------------------------------
def _write_params_to_avr(pf_data, gname: str, params: Dict[str, float]) -> bool:
    """
    Convenience: locate the AVR ElmDsl for *gname* and push the dict.
    Only depends on PowerFactory run‑time API – isolated here.
    """
    # Try to reuse locator from the seeding helper (to stay consistent)
    try:
        import PowerFactory_Interaction.Add_Seed_AVR_Values as SEED
        avr = SEED._locate_avr_block(pf_data, {"name": gname})
    except Exception:
        avr = None

    if avr is None:
        # last‑ditch search
        try:
            blocks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
            for b in blocks:
                if "avr" in b.loc_name.lower() and gname.lower() in b.loc_name.lower():
                    avr = b
                    break
        except Exception:
            pass

    if avr is None:
        print(f"⚠️  AVR block for «{gname}» not found – cannot write params")
        return False

    ok_all = True
    for tag, val in params.items():
        for attr in (f"e:{tag}", tag):
            if avr.SetAttribute(attr, float(val)) == 0:
                break
        else:
            print(f"   ⚠️  could not write {tag} on {avr.loc_name}")
            ok_all = False
    return ok_all
