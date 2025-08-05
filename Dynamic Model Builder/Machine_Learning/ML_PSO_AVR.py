# Machine_Learning/ML_PSO_AVR.py
# ────────────────────────────────────────────────────────────────────────────
#  Lightweight PSO optimiser for AVR parameters (pyswarms‑based)
#  Writes candidates via avr_obj.params list, never via e:<tag> attributes.
#  (single‑particle ask/tell pattern, one RMS run per particle)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import numpy.random as npr
import pyswarms as ps                         # pip install pyswarms
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Tuned parameter list – ORDER must match write_avr_params.py index map
# ---------------------------------------------------------------------------
TUNED_TAGS: List[str] = [
    "Ka", "Ta", "Tr", "Ke", "Te",
    "Kf", "Tf", "Vrmax", "Vrmin"
]
DIM = len(TUNED_TAGS)

# Loose default bounds (all ASCII minus signs)
_BOUNDS_DEF = {
    "Ka":    (10.0, 500.0),
    "Ta":    (1e-3, 0.5),
    "Tr":    (1e-3, 0.2),
    "Ke":    (0.1,  10.0),
    "Te":    (1e-2, 5.0),
    "Kf":    (0.0,  5.0),
    "Tf":    (1e-2, 5.0),
    "Vrmax": (1.0,  20.0),
    "Vrmin": (-20.0, -1.0),
}

# ────────────────────────────────────────────────────────────────────────────
# Helpers – bounds and seed vector
# ────────────────────────────────────────────────────────────────────────────
def _bounds_for(meta: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    user = meta.get("AVR_Bounds", {})
    lb, ub = [], []
    for tag in TUNED_TAGS:
        lo, hi = user.get(tag, _BOUNDS_DEF[tag])
        lb.append(float(lo)); ub.append(float(hi))
    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def _seed_vec(meta: Dict[str, Any],
              lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Return a seed vector **clipped** into [lb, ub]."""
    src = meta.get("AVR_Seed") or meta.get("AVR_Final") or {}
    vec = np.asarray(
        [src.get(t, 0.5*(lo+hi)) for t, lo, hi in zip(TUNED_TAGS, lb, ub)],
        dtype=float,
    )
    return np.clip(vec, lb, ub)                # safeguard

# ────────────────────────────────────────────────────────────────────────────
# PSO state (one per generator)
# ────────────────────────────────────────────────────────────────────────────
class _PSOState:
    def __init__(self,
                 seed: np.ndarray,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 n_particles: int = 10):
        seed = np.clip(seed, lb, ub)
        self.lb, self.ub = lb, ub
        self.n_part      = n_particles
        self.options     = {"c1": 1.5, "c2": 1.5, "w": 0.6}

        rand_part = npr.uniform(lb, ub, size=(n_particles - 1, DIM))
        init_pos  = np.vstack([seed, rand_part])

        self.opt = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=DIM,
            options=self.options,
            bounds=(lb, ub),
            init_pos=init_pos,
        )

        # incremental bookkeeping for ask‑/tell‑one pattern
        self._batch_pos   = self.opt.swarm.position.copy()
        self._batch_costs = np.full(n_particles, np.inf)
        self._next_idx    = 0

    # ---------------- ask / tell ------------------------------------------
    def ask_one(self) -> List[float]:
        if self._next_idx >= self.n_part:               # new generation
            self._batch_pos   = self.opt.swarm.position.copy()
            self._batch_costs = np.full(self.n_part, np.inf)
            self._next_idx    = 0
        vec = self._batch_pos[self._next_idx]
        self._next_idx += 1
        return vec.tolist()

    def tell_one(self, vec: List[float], score: float) -> None:
        idx = self._next_idx - 1                        # particle just scored
        self._batch_costs[idx] = score

        # when all particles of this generation have been evaluated …
        if self._next_idx >= self.n_part:
            sw = self.opt.swarm

            # 0️⃣  first‑generation bootstrap: create pbest arrays
            if sw.pbest_cost.size == 0:
                sw.pbest_cost = self._batch_costs.copy()
                sw.pbest_pos  = self._batch_pos.copy()

            # 1️⃣  update personal bests
            upd = self._batch_costs < sw.pbest_cost
            sw.pbest_cost[upd] = self._batch_costs[upd]
            sw.pbest_pos [upd] = self._batch_pos [upd]

            # 2️⃣  update global best
            best_idx = int(np.argmin(sw.pbest_cost))
            if sw.pbest_cost[best_idx] < sw.best_cost:
                sw.best_cost = float(sw.pbest_cost[best_idx])
                sw.best_pos  = sw.pbest_pos[best_idx].copy()

            # 3️⃣  manual velocity & position update
            r1 = npr.random_sample(size=(self.n_part, DIM))
            r2 = npr.random_sample(size=(self.n_part, DIM))
            c1, c2, w = self.options["c1"], self.options["c2"], self.options["w"]

            sw.velocity = (
                w * sw.velocity
                + c1 * r1 * (sw.pbest_pos - sw.position)
                + c2 * r2 * (sw.best_pos   - sw.position)
            )
            sw.position = np.clip(sw.position + sw.velocity, self.lb, self.ub)
            # next ask_one() starts the new generation

    # helper ---------------------------------------------------------------
    def best_vector(self) -> List[float]:
        return self.opt.swarm.best_pos.tolist()

# ────────────────────────────────────────────────────────────────────────────
# Registry (keyed by generator name)
# ────────────────────────────────────────────────────────────────────────────
_PSO_REG: Dict[str, _PSOState] = {}

def prepare_pso(pf_data, meta: Dict[str, Any], *, n_particles: int = 10):
    g      = meta["name"]
    lb, ub = _bounds_for(meta)
    seed   = _seed_vec(meta, lb, ub)
    _PSO_REG[g] = _PSOState(seed, lb, ub, n_particles)
    print(f"      🐦  PSO ready for «{g}»  (particles={n_particles})")
    return TUNED_TAGS

def ask_one(gname: str) -> List[float]:
    return _PSO_REG[gname].ask_one()

def tell_one(gname: str, score: float, vec: List[float]) -> None:
    _PSO_REG[gname].tell_one(vec, score)

def get_best(gname: str) -> List[float]:
    return _PSO_REG[gname].best_vector()

# ────────────────────────────────────────────────────────────────────────────
# Write vector → PowerFactory AVR block (via params list)
# ────────────────────────────────────────────────────────────────────────────
from PowerFactory_Interaction.write_avr_params import write_avr_params
from PowerFactory_Interaction.Add_Seed_AVR_Values import _locate_avr_block

def _vector_to_param_dict(vec: List[float]) -> Dict[str, float]:
    return {tag: float(vec[i]) for i, tag in enumerate(TUNED_TAGS)}

def write_candidate(pf_data, gname: str, vec: List[float]) -> bool:
    avr = _locate_avr_block(pf_data, {"name": gname})
    if avr is None:
        print(f"⚠️  AVR block for {gname} not found – write skipped")
        return False
    return write_avr_params(
        pf_data,
        avr,
        _vector_to_param_dict(vec),
        verbose=False,
    )
