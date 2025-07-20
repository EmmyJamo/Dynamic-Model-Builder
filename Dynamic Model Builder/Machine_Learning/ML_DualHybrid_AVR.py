"""
ML_DualHybrid_AVR.py
------------------------------------------------------------

Core “dual‑hybrid” optimiser: PSO + CMA‑ES working side‑by‑side,
exposed through a simple ask / tell API.

Used by:
    • gen_tuner.py               (legacy single‑step path)
    • ML_DualHybrid_AVR_wave.py  (two‑candidate wave path)

External deps
-------------
    pip install cma        # real CMA‑ES
    # pyswarms *could* be wired in later; current PSO is a light custom engine
"""

from __future__ import annotations
import math, random, datetime
from typing import Dict, Any, List, Tuple, Optional

# ---------------------------------------------------------------------------
# external library (CMA‑ES)
# ---------------------------------------------------------------------------
import cma                             # type: ignore

# ---------------------------------------------------------------------------
# AVR parameter tags – order is IMPORTANT everywhere
# ---------------------------------------------------------------------------
TUNED_TAGS = ["Ka", "Ta", "Tr", "Ke", "Te", "Kf",
              "Tf", "Vrmax", "Vrmin"]           # 9 parameters

_EXTRA_TAGS = ["E1", "E2", "Se1", "Se2"]

# ---------------------------------------------------------------------------
# loose default bounds (can be overridden per‑machine)
# ---------------------------------------------------------------------------
# Parameter bounds (all ASCII minus signs)
# ---------------------------------------------------------------------------
_BOUNDS_DEFAULT: dict[str, tuple[float, float]] = {
    "Ka":    (10.0, 500.0),
    "Ta":    (1e-3,  0.5),
    "Tr":    (1e-3,  0.2),
    "Ke":    (0.1,   10.0),
    "Te":    (1e-2,  5.0),
    "Kf":    (0.0,   5.0),
    "Tf":    (1e-2,  5.0),
    "Vrmax": (1.0,   20.0),
    "Vrmin": (-20.0, -1.0),
}

# ---------------------------------------------------------------------------
# PSO engine  (simple, ask/tell friendly)
# ---------------------------------------------------------------------------
from typing import List
import random

class _PSO:
    def __init__(
        self,
        seed: List[float],
        lb: List[float],
        ub: List[float],
        n_part: int = 10,
        w: float = 0.6,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        self.dim = len(seed)
        self.lb, self.ub = lb, ub
        self.w, self.c1, self.c2 = w, c1, c2

        # particle positions (first = seed, rest random)
        self.pos: List[List[float]] = [seed[:]] + [
            [random.uniform(lb[i], ub[i]) for i in range(self.dim)]
            for _ in range(n_part - 1)      # <-- ASCII minus here
        ]
        self.vel: List[List[float]] = [[0.0] * self.dim for _ in range(n_part)]
        self.pbest = self.pos[:]
        self.pbest_f = [float("inf")] * n_part
        self.gbest = seed[:]
        self.gbest_f = float("inf")

    # ------------------------------------------------------------------ ask
    def ask(self) -> List[List[float]]:
        if not self._queue:
            # deliver whole swarm once per generation
            self._queue = [p[:] for p in self.pos]
        return [self._queue.pop(0)]

    # ----------------------------------------------------------------- tell
    def tell(self, xs: List[List[float]], fs: List[float]):
        for x, f in zip(xs, fs):
            # which particle?  – find nearest index in current generation
            try:
                idx = self.pos.index(x)
            except ValueError:
                # fallback: first unevaluated
                idx = self.pbest_f.index(math.inf)
            self.pos[idx] = x[:]

            if f < self.pbest_f[idx]:
                self.pbest_f[idx] = f
                self.pbest[idx]  = x[:]
            if f < self.gbest_f:
                self.gbest_f, self.gbest = f, x[:]

        if not self._queue:              # generation finished → update step
            for i in range(len(self.pos)):
                for d in range(self.dim):
                    r1, r2 = random.random(), random.random()
                    cog = self.c1*r1*(self.pbest[i][d]-self.pos[i][d])
                    soc = self.c2*r2*(self.gbest[d]-self.pos[i][d])
                    self.vel[i][d] = self.w*self.vel[i][d] + cog + soc
                    self.pos[i][d] += self.vel[i][d]
                    # clamp
                    if self.pos[i][d] < self.lb[d]:
                        self.pos[i][d] = self.lb[d]; self.vel[i][d]=0
                    elif self.pos[i][d] > self.ub[d]:
                        self.pos[i][d] = self.ub[d]; self.vel[i][d]=0

# ---------------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────
#  CMA‑ES wrapper  (real library)
# ────────────────────────────────────────────────────────────────────────────
class _CMA:
    def __init__(self, seed: List[float], lb: List[float], ub: List[float],
                 sigma_frac:float = 0.25, lam:int = 6):
        self.lb, self.ub = lb, ub
        sig0 = sigma_frac * sum(ub[i]-lb[i] for i in range(len(lb))) / len(lb)
        opts = {"bounds":[lb, ub], "popsize": lam, "verb_log":0, "verb_disp":0}
        self.es = cma.CMAEvolutionStrategy(seed, sig0, opts)
        self._asked : List[List[float]] = []

    def ask(self) -> List[List[float]]:
        if not self._asked:
            self._asked = self.es.ask()
        return [self._asked.pop(0)]

    def tell(self, xs: List[List[float]], fs: List[float]):
        # accumulate until we have a full λ batch
        if not hasattr(self, "_acc_x"): self._acc_x, self._acc_f = [], []
        self._acc_x += xs
        self._acc_f += fs
        if len(self._acc_x) >= self.es.popsize:
            self.es.tell(self._acc_x, self._acc_f)
            self._acc_x, self._acc_f = [], []

    # convenience
    @property
    def best_f(self): return self.es.best.f
    @property
    def best_x(self): return self.es.best.x

# ---------------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────
#  Hybrid controller (PSO + CMA)
# ────────────────────────────────────────────────────────────────────────────
class _Hybrid:
    def __init__(self, seed:List[float], lb:List[float], ub:List[float]):
        self.pso = _PSO(seed, lb, ub)
        self.cma = _CMA(seed, lb, ub)
        self.credit_cma = 0.5        # share of iterations to CMA
        self.min_credit  = 0.2
        self.best_x, self.best_f = seed[:], math.inf
        self.iter = 0
        self._last_scenario = "fast_dip"   # default

    # ------------------------------------------------------------ one step
    def ask_two(self) -> Tuple[List[float], List[float]]:
        """Return (vec_from_pso, vec_from_cma)."""
        vec_pso = self.pso.ask()[0]
        vec_cma = self.cma.ask()[0]
        return vec_pso, vec_cma

    def _update_credit(self, src:str, f:float):
        if f < self.best_f:
            self.best_f, self.best_x = f, (self.pso.gbest if src=="PSO" else self.cma.best_x)[:]
            # reward engine
            if src=="CMA":
                self.credit_cma = min(0.9, self.credit_cma+0.05)
            else:
                self.credit_cma = max(self.min_credit, self.credit_cma-0.05)

    # ----------------------------------------------------------- feedback
    def tell(self, src:str, vec:List[float], score:float):
        if src=="PSO":
            self.pso.tell([vec],[score])
        else:
            self.cma.tell([vec],[score])
        self._update_credit(src, score)

    # ------------------------------------------------------------ helpers
    def maybe_cross_seed(self, period:int =5):
        self.iter += 1
        if self.iter % period == 0:
            self.pso.gbest = self.best_x[:]
            self.cma.es.set_xmean(self.best_x[:])

# ---------------------------------------------------------------------------
# GLOBAL REGISTRY  (one Hybrid per generator)
# ---------------------------------------------------------------------------
_HYBRID_REG : Dict[str, _Hybrid] = {}

def _bounds_for(meta:Dict[str,Any]) -> Tuple[List[float],List[float]]:
    lb, ub = [], []
    user = meta.get("AVR_Bounds", {})
    for tag in TUNED_TAGS:
        lo, hi = user.get(tag, _BOUNDS_DEFAULT[tag])
        lb.append(lo); ub.append(hi)
    return lb, ub

def _seed_vec(meta:Dict[str,Any], lb,ub) -> List[float]:
    src = meta.get("AVR_Seed") or meta.get("AVR_Final") or {}
    return [src.get(tag, 0.5*(lo+hi)) for tag,(lo,hi) in zip(TUNED_TAGS, zip(lb,ub))]

def _get_hybrid(meta:Dict[str,Any]) -> _Hybrid:
    g = meta["name"]
    if g in _HYBRID_REG: return _HYBRID_REG[g]
    lb,ub = _bounds_for(meta)
    seed  = _seed_vec(meta, lb, ub)
    _HYBRID_REG[g] = _Hybrid(seed, lb, ub)
    return _HYBRID_REG[g]

# ---------------------------------------------------------------------------
# Public helpers used by the wave layer
# ---------------------------------------------------------------------------
def _ask_from_pso(pso:_PSO) -> List[float]:
    return pso.ask()[0]

def _ask_from_cma(cma:_CMA) -> List[float]:
    return cma.ask()[0]

def _feedback_score(gname:str, src:str, score:float):
    h = _HYBRID_REG[gname]
    if src=="PSO":
        vec = h.pso.pos[h.pso.pos.index(h.pso.gbest)] if h.pso.pos else h.best_x
        h.tell("PSO", vec, score)
    else:
        vec = h.cma.best_x
        h.tell("CMA", vec, score)

def _update_credit(h:_Hybrid, src:str, score:float):
    h._update_credit(src, score)

def _maybe_cross_seed(h:_Hybrid):
    h.maybe_cross_seed()

def get_best_so_far(gname:str) -> List[float]:
    return _HYBRID_REG[gname].best_x[:]

# ---------------------------------------------------------------------------
# Write parameters into PF AVR block  (used by wave + tuner wrapper)
# ---------------------------------------------------------------------------
def _write_params_to_avr(pf_data, gname:str, params:Dict[str,float]) -> bool:
    # locate AVR  (reuse location helper from seeding module – keeps import‑free)
    try:
        import PowerFactory_Interaction.Add_Seed_AVR_Values as SEED
        avr = SEED._locate_avr_block(pf_data, {"name":gname})
    except Exception:
        # emergency crawl
        avr = None
        try:
            blocks = pf_data.app.GetCalcRelevantObjects("*.ElmDsl")
            for b in blocks:
                if "avr" in b.loc_name.lower() and gname.lower() in b.loc_name.lower():
                    avr = b; break
        except Exception:
            pass
    if avr is None:
        print(f"⚠️  write_params_to_avr: AVR block for {gname} not found"); return False
    ok = True
    for tag,val in params.items():
        for attr in (f"e:{tag}", tag):
            if avr.SetAttribute(attr, float(val)) == 0: break
        else:
            print(f"   ⚠️ could not write {tag}")
            ok=False

    return ok