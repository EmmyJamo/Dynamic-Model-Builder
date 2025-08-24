# Machine_Learning/ML_PSO_AVR.py
# ────────────────────────────────────────────────────────────────────────────
#  Lightweight PSO optimiser for AVR parameters (pyswarms-based)
#  Single-particle ask/tell pattern (one RMS run per particle).
#  Writes candidates to PowerFactory via the AVR .params vector.
#
#  Public API:
#     prepare_pso(pf_data, meta, *, n_particles=10) -> List[str]
#     ask_one(gname)                                 -> List[float]
#     tell_one(gname, score, vec)                    -> None
#     get_best(gname)                                -> List[float]
#     bind_avr(pf_data, meta)                        -> bool
#     write_candidate(pf_data, meta, vec)            -> bool
#
#  Diagnostics / Visualisation:
#     save_history_csv(gname, out_dir)               -> Path
#     export_heatmaps(gname, out_dir, ...)           -> List[Path]
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import numpy.random as npr
import pyswarms as ps                     # pip install pyswarms
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# plotting (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Tuned parameter list – keep order stable and consistent with writer
# ---------------------------------------------------------------------------
TUNED_TAGS: List[str] = [
    "Ka", "Ta", "Tr", "Ke", "Te",
    "Kf", "Tf", "Vrmax", "Vrmin",
]
DIM = len(TUNED_TAGS)

# Loose default bounds (ASCII minus)
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

# Default PSO + anti-stagnation behaviour (overridable via meta["PSO_Options"])
_PSO_DEFAULTS = dict(
    c1=1.5, c2=1.5, w=0.6,                      # base options
    vmax_frac=0.25,                              # clamp |v| <= frac*(ub-lb)
    jitter0=0.05, jitter_decay=0.92,             # Gaussian jitter (% of span)
    plateau_gens=3,                              # generations with no-improve to trigger re-injection
    tol_abs=2e-4, tol_rel=0.01,                  # improvement thresholds
    reinject_frac=0.20,                          # replace worst 20% on plateau
    explore_frac_init=0.60,                      # radius (as % of span) around gbest for reinjection
    explore_frac_floor=0.20,                     # minimum radius as it anneals
    w_bounds=(0.35, 0.90),                       # inertia bounds
    w_up=1.05, w_down=0.99,                      # how quickly w adapts
    seed=None,                                   # reproducibility (int|None)
)

# ────────────────────────────────────────────────────────────────────────────
# Helpers – bounds and seed vector
# ────────────────────────────────────────────────────────────────────────────
def _bounds_for(meta: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    user = meta.get("AVR_Bounds", {})
    lb, ub = [], []
    for tag in TUNED_TAGS:
        lo, hi = user.get(tag, _BOUNDS_DEF[tag])
        lb.append(float(lo))
        ub.append(float(hi))
    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

def _seed_vec(meta: Dict[str, Any],
              lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Return a seed vector clipped into [lb, ub]."""
    src = meta.get("AVR_Seed") or meta.get("AVR_Final") or {}
    vec = np.asarray(
        [src.get(t, 0.5*(lo+hi)) for t, lo, hi in zip(TUNED_TAGS, lb, ub)],
        dtype=float,
    )
    return np.clip(vec, lb, ub)

def _merge_opts(meta: Dict[str, Any]) -> dict:
    opts = _PSO_DEFAULTS.copy()
    user = meta.get("PSO_Options") or {}
    for k, v in user.items():
        if k in opts:
            opts[k] = v
    # allow explicit reproducible seed in meta["PSO_Seed"]
    if meta.get("PSO_Seed") is not None:
        opts["seed"] = int(meta["PSO_Seed"])
    return opts

def _vec_to_named(vec: List[float] | np.ndarray) -> Dict[str, float]:
    arr = np.asarray(vec, dtype=float).ravel()
    return {tag: float(arr[i]) for i, tag in enumerate(TUNED_TAGS)}

# ────────────────────────────────────────────────────────────────────────────
# Utility samplers
# ────────────────────────────────────────────────────────────────────────────
def _lhs_unit(n: int, d: int, rng) -> np.ndarray:
    """Simple Latin Hypercube sample in [0,1]^d using a RNG with .random()."""
    # stratify each dim into n bins, then permute bins per dim
    u = (np.arange(n) + rng.random(n)) / n
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        out[:, j] = rng.permutation(u)
    return out

def _reflect_into_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Reflect positions that go out of bounds back into [lb,ub].
    Works for both 1-D vectors (DIM,) and 2-D matrices (N, DIM).
    """
    x_ref = np.asarray(x, dtype=float).copy()

    # Broadcast lb/ub to match x's ndim
    if x_ref.ndim == 1:
        lb_b = np.asarray(lb, dtype=float)
        ub_b = np.asarray(ub, dtype=float)
    else:
        lb_b = np.asarray(lb, dtype=float).reshape(1, -1)
        ub_b = np.asarray(ub, dtype=float).reshape(1, -1)

    # Guard zero-span dims to avoid modulo by zero
    span = np.maximum(ub_b - lb_b, 1e-12)

    # First reflect values below/above, then clamp for safety
    x_ref = np.where(
        x_ref < lb_b,
        lb_b + (lb_b - x_ref) % (2.0 * span),
        x_ref,
    )
    x_ref = np.where(
        x_ref > ub_b,
        ub_b - (x_ref - ub_b) % (2.0 * span),
        x_ref,
    )

    return np.minimum(np.maximum(x_ref, lb_b), ub_b)

# ────────────────────────────────────────────────────────────────────────────
# Internal history store for diagnostics
# ────────────────────────────────────────────────────────────────────────────
# Per-generator list of dicts:
# { "gen": int, "particle": int, "score": float, "<param>": value, ... }
_HIST: Dict[str, List[Dict[str, float]]] = {}

def _hist_append(gname: str, generation: int, particle_idx: int,
                 vec: List[float] | np.ndarray, score: float) -> None:
    rec = {"gen": int(generation), "particle": int(particle_idx), "score": float(score)}
    rec.update(_vec_to_named(vec))
    _HIST.setdefault(gname, []).append(rec)

def get_history(gname: str) -> List[Dict[str, float]]:
    """Return the in-memory candidate history for a generator."""
    return list(_HIST.get(gname, []))

def save_history_csv(gname: str, out_dir: Path | str) -> Path:
    """Persist the evaluation history (one row per candidate)."""
    import csv
    rows = _HIST.get(gname, [])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{gname}_history.csv"
    if not rows:
        # still create header from TUNED_TAGS
        with p.open("w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["gen", "particle", "score"] + TUNED_TAGS)
        return p
    # ensure stable column order
    cols = ["gen", "particle", "score"] + TUNED_TAGS
    with p.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"   📝 PSO history saved → {p}")
    return p

def export_heatmaps(gname: str,
                    out_dir: Path | str,
                    *,
                    pairs: Optional[List[Tuple[str, str]]] = None,
                    bins: int = 36,
                    use_log10: bool = True) -> List[Path]:
    """
    Create density heatmaps for selected parameter pairs, coloured by best (low) scores.
    """
    out_dir = Path(out_dir)
    heat_dir = out_dir
    heat_dir.mkdir(parents=True, exist_ok=True)
    rows = _HIST.get(gname, [])
    if not rows:
        print(f"   (no history for «{gname}», skipping heatmaps)")
        return []
    # default pairs
    if not pairs:
        pairs = [("Ka","Ta"), ("Ka","Kf"), ("Ke","Te"), ("Vrmax","Vrmin"), ("Ta","Tr")]

    # pull arrays
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    if use_log10:
        # guard zeros
        s = np.log10(np.maximum(scores, 1e-12))
    else:
        s = scores.copy()

    out_paths: List[Path] = []
    for (x_tag, y_tag) in pairs:
        x = np.asarray([r.get(x_tag, np.nan) for r in rows], dtype=float)
        y = np.asarray([r.get(y_tag, np.nan) for r in rows], dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
        if np.count_nonzero(m) < 10:
            continue
        x, y, s_use = x[m], y[m], s[m]

        # Build a 2D grid and take the MIN score per cell (best)
        H_count, xedges, yedges = np.histogram2d(x, y, bins=bins)
        H_best = np.full_like(H_count, np.nan, dtype=float)

        # assign each point to a bin, track min score
        xi = np.clip(np.digitize(x, xedges) - 1, 0, H_best.shape[0]-1)
        yi = np.clip(np.digitize(y, yedges) - 1, 0, H_best.shape[1]-1)
        for i in range(len(xi)):
            xb, yb = xi[i], yi[i]
            val = s_use[i]
            if np.isnan(H_best[xb, yb]) or val < H_best[xb, yb]:
                H_best[xb, yb] = val

        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        # show best score (log10 if requested); invert colormap so lower=better (darker)
        im = ax.imshow(
            np.flipud(H_best.T),  # orient bins to standard origin
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis_r"
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("log10(fitness)" if use_log10 else "fitness")
        ax.set_xlabel(x_tag)
        ax.set_ylabel(y_tag)
        ax.set_title(f"{gname}: best score in each cell")
        fig.tight_layout()
        out_p = heat_dir / f"{gname}_{x_tag}_vs_{y_tag}.png"
        fig.savefig(out_p, dpi=150)
        plt.close(fig)
        out_paths.append(out_p)
        print(f"   📈 heatmap saved → {out_p}")
    return out_paths

# ────────────────────────────────────────────────────────────────────────────
# PSO state (one per generator) – manual step to avoid calling opt.step()
# with anti-stagnation logic and adaptive inertia
# ────────────────────────────────────────────────────────────────────────────
class _PSOState:
    def __init__(self,
                 seed: np.ndarray,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 n_particles: int = 20,
                 opts: dict | None = None):
        opts = (opts or {}).copy()
        self.lb, self.ub = lb, ub
        self.n_part      = n_particles

        # options / behaviour knobs
        self.c1     = float(opts.get("c1", _PSO_DEFAULTS["c1"]))
        self.c2     = float(opts.get("c2", _PSO_DEFAULTS["c2"]))
        self.w      = float(opts.get("w",  _PSO_DEFAULTS["w"]))
        self.vmax_f = float(opts.get("vmax_frac", _PSO_DEFAULTS["vmax_frac"]))
        self.j0     = float(opts.get("jitter0", _PSO_DEFAULTS["jitter0"]))
        self.jdec   = float(opts.get("jitter_decay", _PSO_DEFAULTS["jitter_decay"]))
        self.plat_g = int(opts.get("plateau_gens", _PSO_DEFAULTS["plateau_gens"]))
        self.tol_abs= float(opts.get("tol_abs", _PSO_DEFAULTS["tol_abs"]))
        self.tol_rel= float(opts.get("tol_rel", _PSO_DEFAULTS["tol_rel"]))
        self.reinj_f= float(opts.get("reinject_frac", _PSO_DEFAULTS["reinject_frac"]))
        self.expl_f = float(opts.get("explore_frac_init", _PSO_DEFAULTS["explore_frac_init"]))
        self.expl_floor = float(opts.get("explore_frac_floor", _PSO_DEFAULTS["explore_frac_floor"]))
        self.w_min, self.w_max = opts.get("w_bounds", _PSO_DEFAULTS["w_bounds"])
        self.w_up   = float(opts.get("w_up", _PSO_DEFAULTS["w_up"]))
        self.w_down = float(opts.get("w_down", _PSO_DEFAULTS["w_down"]))

        # reproducibility
        if opts.get("seed") is not None:
            npr.seed(int(opts["seed"]))

        seed = np.clip(seed, lb, ub)
        span = (ub - lb)

        # initial swarm (1 seed + LHS randoms within bounds)
        if n_particles > 1:
            unit = _lhs_unit(n_particles - 1, DIM, npr)
            rand_part = lb + unit * span
            init_pos  = np.vstack([seed, rand_part])
        else:
            init_pos  = seed.reshape(1, -1)

        self.options = {"c1": self.c1, "c2": self.c2, "w": self.w}

        self.opt = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=DIM,
            options=self.options,
            bounds=(lb, ub),
            init_pos=init_pos,
        )

        # incremental bookkeeping for ask/tell-one
        self._batch_pos   = self.opt.swarm.position.copy()
        self._batch_costs = np.full(n_particles, np.inf)
        self._next_idx    = 0

        # Ensure swarm has velocity arrays
        sw = self.opt.swarm
        if sw.velocity is None or sw.velocity.shape != sw.position.shape:
            sw.velocity = np.zeros_like(sw.position)

        # anti-stagnation state
        self.vmax     = self.vmax_f * (ub - lb)
        self.generation = 0
        self.best_hist: List[float] = []
        self.no_improve_gens = 0
        self.jitter_scale = self.j0  # annealed

    # ---------------- ask / tell ------------------------------------------
    def ask_one(self) -> List[float]:
        if self._next_idx >= self.n_part:               # new generation
            self._batch_pos   = self.opt.swarm.position.copy()
            self._batch_costs = np.full(self.n_part, np.inf)
            self._next_idx    = 0
        vec = self._batch_pos[self._next_idx]
        self._next_idx += 1
        return vec.tolist()

    def _maybe_adapt_inertia(self, improved: bool):
        # if we improved, nudge w down (exploit); else up (explore)
        if improved:
            self.w = max(self.w_min, min(self.w_max, self.w * self.w_down))
        else:
            self.w = max(self.w_min, min(self.w_max, self.w * self.w_up))
        # keep pyswarms option in sync
        self.opt.options["w"] = self.w

    def _plateau_triggered(self, prev_best: float, new_best: float) -> bool:
        # improvement threshold: relative OR absolute
        if not np.isfinite(prev_best):
            return False
        delta = prev_best - new_best
        thresh = max(self.tol_abs, self.tol_rel * abs(prev_best))
        return delta <= thresh

    def _reinject_diversity(self, sw):
        """Replace worst K particles around (or away from) gbest."""
        K = max(1, int(self.reinj_f * self.n_part))
        # indices of worst K by pbest_cost (or batch costs if first gen)
        if getattr(sw, "pbest_cost", None) is not None and sw.pbest_cost.size:
            order = np.argsort(sw.pbest_cost)[::-1]  # descending (worst first)
        else:
            order = np.argsort(self._batch_costs)[::-1]
        worst_idx = order[:K]

        gbest = sw.best_pos.copy()
        span  = (self.ub - self.lb)
        radius = np.maximum(self.expl_floor, self.expl_f) * span
        # sample around gbest within a box, then reflect into bounds
        noise = (npr.random((K, DIM)) - 0.5) * 2.0 * radius
        new_pos = _reflect_into_bounds(gbest + noise, self.lb, self.ub)
        sw.position[worst_idx, :] = new_pos

        # reset their velocities to small random values
        sw.velocity[worst_idx, :] = (npr.uniform(-0.1, 0.1, size=(K, DIM)) * span)

        # slowly anneal explore radius & jitter
        self.expl_f = max(self.expl_floor, self.expl_f * 0.9)
        self.jitter_scale *= self.jdec

    def tell_one(self, vec: List[float], score: float) -> None:
        idx = self._next_idx - 1                        # particle just scored
        self._batch_costs[idx] = float(score)

        # record this evaluation in diagnostics history
        _hist_append(self.opt.swarm.pbest_pos.shape[0] and getattr(self.opt, "name", "gen") or "gen",
                     self.generation, idx, vec, score)  # gname isn’t known here; wrapper logs by public API below

        # When all particles in the generation have been evaluated …
        if self._next_idx >= self.n_part:
            sw = self.opt.swarm

            # Bootstrap pbest arrays on first pass (older pyswarms versions)
            if getattr(sw, "pbest_cost", None) is None or sw.pbest_cost.size == 0:
                sw.pbest_cost = self._batch_costs.copy()
                sw.pbest_pos  = self._batch_pos.copy()
                best_idx = int(np.argmin(sw.pbest_cost))
                sw.best_cost = float(sw.pbest_cost[best_idx])
                sw.best_pos  = sw.pbest_pos[best_idx].copy()
                prev_best = np.inf
            else:
                prev_best = float(sw.best_cost)
                # Update personal bests
                upd = self._batch_costs < sw.pbest_cost
                if np.any(upd):
                    sw.pbest_cost[upd] = self._batch_costs[upd]
                    sw.pbest_pos [upd] = self._batch_pos [upd]

                # Update global best
                best_idx = int(np.argmin(sw.pbest_cost))
                if sw.pbest_cost[best_idx] < sw.best_cost:
                    sw.best_cost = float(sw.pbest_cost[best_idx])
                    sw.best_pos  = sw.pbest_pos[best_idx].copy()

            # Adapt inertia & track plateau
            improved = float(sw.best_cost) < prev_best
            self._maybe_adapt_inertia(improved)
            if self._plateau_triggered(prev_best, float(sw.best_cost)):
                self.no_improve_gens += 1
            else:
                self.no_improve_gens = 0

            # Manual velocity & position update (no call to opt.step)
            r1 = npr.random_sample(size=(self.n_part, DIM))
            r2 = npr.random_sample(size=(self.n_part, DIM))
            c1, c2, w = self.c1, self.c2, self.w

            # Gaussian jitter (annealed) promotes exploration
            jitter = self.jitter_scale * (self.ub - self.lb) * npr.normal(size=sw.position.shape)

            sw.velocity = (
                w * sw.velocity
                + c1 * r1 * (sw.pbest_pos - sw.position)
                + c2 * r2 * (sw.best_pos   - sw.position)
            )

            # clamp velocities
            sw.velocity = np.clip(sw.velocity, -self.vmax, self.vmax)

            # position update + jitter, then reflect into bounds
            new_pos = sw.position + sw.velocity + jitter
            sw.position = _reflect_into_bounds(new_pos, self.lb, self.ub)

            # plateau-based diversity injection
            if self.no_improve_gens >= self.plat_g and self.n_part > 2:
                self._reinject_diversity(sw)
                self.no_improve_gens = 0  # reset after reinjection

            # next ask_one() begins a new generation
            self.generation += 1
            self.best_hist.append(float(sw.best_cost))

    # helper ---------------------------------------------------------------
    def best_vector(self) -> List[float]:
        return self.opt.swarm.best_pos.tolist()

# ────────────────────────────────────────────────────────────────────────────
# Registry (keyed by generator name) + public API
# ────────────────────────────────────────────────────────────────────────────
_PSO_REG: Dict[str, _PSOState] = {}
# Map gname -> canonical name for diagnostics storage
_NAME_MAP: Dict[str, str] = {}

def prepare_pso(pf_data, meta: Dict[str, Any], *, n_particles: int = 10) -> List[str]:
    g      = meta["name"]
    lb, ub = _bounds_for(meta)
    seed   = _seed_vec(meta, lb, ub)
    opts   = _merge_opts(meta)
    state  = _PSOState(seed, lb, ub, n_particles, opts)
    _PSO_REG[g] = state
    _NAME_MAP[g] = g  # stable key into _HIST
    print(f"      🐦  PSO ready for «{g}»  (particles={n_particles})")
    return TUNED_TAGS

def ask_one(gname: str) -> List[float]:
    return _PSO_REG[gname].ask_one()

def tell_one(gname: str, score: float, vec: List[float]) -> None:
    # Append to the correct generator history with its real name
    _hist_append(_NAME_MAP.get(gname, gname),
                 _PSO_REG[gname].generation,
                 _PSO_REG[gname]._next_idx - 1 if _PSO_REG[gname]._next_idx > 0 else 0,
                 vec, score)
    _PSO_REG[gname].tell_one(vec, score)

def get_best(gname: str) -> List[float]:
    return _PSO_REG[gname].best_vector()

# Convenience passthroughs for diagnostics
def save_history_csv_public(gname: str, out_dir: Path | str) -> Path:
    return save_history_csv(_NAME_MAP.get(gname, gname), out_dir)

def export_heatmaps_public(gname: str, out_dir: Path | str,
                           *, pairs: Optional[List[Tuple[str, str]]] = None,
                           bins: int = 36, use_log10: bool = True) -> List[Path]:
    return export_heatmaps(_NAME_MAP.get(gname, gname), out_dir,
                           pairs=pairs, bins=bins, use_log10=use_log10)

# ────────────────────────────────────────────────────────────────────────────
# Write vector → PowerFactory AVR block (via params list), with cached handle
# ────────────────────────────────────────────────────────────────────────────
from PowerFactory_Interaction.write_avr_params import write_avr_params
from PowerFactory_Interaction.Add_Seed_AVR_Values import _locate_avr_block

# Cache of AVR ElmDsl handles, keyed by generator name
_AVR_HANDLE: Dict[str, object] = {}

def bind_avr(pf_data, meta: Dict[str, Any]) -> bool:
    """
    Locate and cache the AVR ElmDsl object for this generator (uses AVR_Name if present).
    Call this once after seeding and before optimisation runs.
    """
    gname = meta["name"]
    try:
        avr = _locate_avr_block(pf_data, meta)  # consistent with seeding logic
    except Exception as e:
        print(f"⚠️  bind_avr: locate failed for «{gname}»: {e}")
        return False
    if avr is None:
        print(f"⚠️  bind_avr: AVR block not found for «{gname}»")
        return False
    _AVR_HANDLE[gname] = avr
    return True

def _vector_to_param_dict(vec: List[float]) -> Dict[str, float]:
    return {tag: float(vec[i]) for i, tag in enumerate(TUNED_TAGS)}

def write_candidate(pf_data, meta: Dict[str, Any], vec: List[float]) -> bool:
    """
    Write vector into the SAME AVR ElmDsl that was bound/seeded.
    Falls back to rebinding if needed.
    """
    gname = meta["name"]
    avr = _AVR_HANDLE.get(gname)
    if avr is None:
        if not bind_avr(pf_data, meta):
            print(f"⚠️  write_candidate: AVR not bound for «{gname}» – write skipped")
            return False
        avr = _AVR_HANDLE[gname]

    return write_avr_params(
        pf_data,
        avr,
        _vector_to_param_dict(vec),
        verbose=False,
    )

def get_bound_avr(gname: str):
    """Optional debug helper: return cached AVR handle or None."""
    return _AVR_HANDLE.get(gname)

# Aliases the wrapper can call (friendlier names)
save_history_csv = save_history_csv_public
export_heatmaps  = export_heatmaps_public
