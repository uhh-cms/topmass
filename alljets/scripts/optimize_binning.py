# coding: utf-8

"""
Adaptive (tree) binning of the top-mass measurement in mtfit x m_W,avg (2D) and mtfit x m_W,avg x R_bq (3D).

Input is a datacard of the inference model fine_3D (one config, created with the unrolling hist hook), whose
shapes contain all processes and systematics on the fine grid of alljets/config/binning_tree.py. Starting from one
box covering the grid, the box split (at a fine-grid edge, along any axis) that lowers the expected uncertainty on
the top mass the most is accepted, until --max-leaves or until no split passes the minimum-content and minimum-gain
requirements. The expected uncertainty is computed from the fisher information of an asimov fit with:
  - the top mass as parameter of interest, with the template derivative (up - down) / (2 delta) of --mt-param,
  - one unit-gaussian nuisance per shape and lnN parameter (sum of the effects over the processes, --exclude
    parameters skipped), optionally unconstrained normalizations (--float-norm),
  - the per-bin MC statistical variance added to the expected yield (as for autoMCStats, Barlow-Beeston lite),
  - the noise of the derivative (independent top-mass samples) subtracted from its fisher information.
The trajectory is evaluated on the datacard of another year (--test-card, independent MC), and the number of
leaves with the smallest test uncertainty is written to alljets/config/binning_tree_{2D,3D}.json.
With several --train-card datacards (e.g. one per year), the tree is grown with the summed fisher information of a
joint fit: nuisances with the same name are correlated, the normalizations of --float-norm are separate per
datacard, and the minimum-content requirements hold in each datacard. Use --max-leaves or --leaves to fix the size.

Example:
  python alljets/scripts/optimize_binning.py --train-card <2017 datacard.txt> --test-card <2018 datacard.txt>
  python alljets/scripts/optimize_binning.py --train-card <2017 datacard.txt> <2018 datacard.txt> --max-leaves 35
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import logging
import os
import re

import numpy as np
import uproot

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("optimize_binning")

# load binning_tree.py directly, importing the alljets package would require columnflow
_spec = importlib.util.spec_from_file_location(
    "binning_tree",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "binning_tree.py"),
)
binning_tree = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(binning_tree)

# rows of the per-cell quantities, followed by the derivative of the top mass and the nuisance effects
NU, VAR_MC, SIG, VAR_DMT, VEC = 0, 1, 2, 3, 4


def parse_card(path: str, category: str | None):
    # shapes file, category, processes (with signal flags) and parameters (type, {process: effect}) of a datacard
    rows = [
        line.split() for line in open(path)
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]
    shapes = next(r[3] for r in rows if r[0] == "shapes")
    start = next(i for i, r in enumerate(rows) if r[0] == "bin" and rows[i + 1][0] == "process")
    bins, procs, ids = rows[start][1:], rows[start + 1][1:], rows[start + 2][1:]
    category = category or bins[0]
    cols = [k for k, b in enumerate(bins) if b == category]
    if not cols:
        raise ValueError(f"category {category} not in datacard {path}, found {sorted(set(bins))}")
    params = {}
    for r in rows[start + 4:]:
        if len(r) < 2 + len(bins):
            continue
        if r[1] not in ("shape", "shapeN", "shape?", "lnN"):
            if r[1] not in ("autoMCStats", "group", "="):
                logger.warning(f"ignoring datacard line: {' '.join(r)}")
            continue
        effects = {procs[k]: r[2 + k] for k in cols if r[2 + k] != "-"}
        if effects:
            params[r[0]] = (r[1], effects)
    shapes = os.path.join(os.path.dirname(os.path.abspath(path)), shapes)
    return shapes, category, [procs[k] for k in cols], {procs[k]: int(ids[k]) <= 0 for k in cols}, params


class Model:
    """
    Per-cell quantities of a fine-grid datacard as prefix sums, so that the sums over any box are cheap.
    """

    def __init__(self, card: str, args: argparse.Namespace, dims: int, tag: str = ""):
        shapes, category, procs, signal, params = parse_card(card, args.category)
        self.card = card
        axes = binning_tree.FINE_AXES
        shape3 = tuple(len(binning_tree.FINE_EDGES[a]) - 1 for a in axes)
        f = uproot.open(shapes)

        def read(key):
            h = f[f"{category}/{key}"]
            values = np.asarray(h.values(flow=False), dtype=np.float64)
            variances = h.variances(flow=False)
            variances = np.zeros_like(values) if variances is None else np.asarray(variances, dtype=np.float64)
            if values.size != np.prod(shape3):
                raise ValueError(f"{key} has {values.size} bins, the fine grid {shape3} has {np.prod(shape3)}")
            values, variances = values.reshape(shape3), variances.reshape(shape3)
            # 2D: sum over the R_bq axis
            if dims == 2:
                values, variances = values.sum(axis=2), variances.sum(axis=2)
            return values, variances

        nom = {p: read(p) for p in procs}
        grid = next(iter(nom.values()))[0].shape
        nu = sum(v for v, _ in nom.values())
        var_mc = sum(w for _, w in nom.values())
        sig = sum(nom[p][0] for p in procs if signal[p])

        def shape_delta(param, proc, scale):
            up, var_up = read(f"{proc}__{param}Up")
            down, var_down = read(f"{proc}__{param}Down")
            return scale * 0.5 * (up - down), scale ** 2 * 0.25 * (var_up + var_down)

        # top-mass dependence per bin: least-squares pol2 in the mass shift through the nominal template and all
        # variations, the derivative at the nominal mass is the linear coefficient. The points are weighted with
        # the inverse effective event weight of their sample (constant over the bins), so the fit is a fixed
        # linear combination of the templates and its variance follows from their sumw2.
        mt_params = args.mt_param.split(",")
        mt_deltas = [float(x) for x in str(args.mt_delta).split(",")]
        if len(mt_params) != len(mt_deltas):
            raise ValueError(f"--mt-param {args.mt_param} and --mt-delta {args.mt_delta} differ in length")
        missing = [p for p in mt_params if p not in params]
        if missing:
            raise ValueError(f"top-mass parameters {missing} not in datacard {card}")
        mt_procs = sorted({proc for param in mt_params for proc in params[param][1]})
        xs = [0.0]
        ys = [sum(nom[proc][0] for proc in mt_procs)]
        vs = [sum(nom[proc][1] for proc in mt_procs)]
        for param, k in zip(mt_params, mt_deltas):
            for sign, direction in ((1.0, "Up"), (-1.0, "Down")):
                templates = [read(f"{proc}__{param}{direction}") for proc in mt_procs]
                xs.append(sign * k)
                ys.append(sum(t[0] for t in templates))
                vs.append(sum(t[1] for t in templates))
        x, y, var_y = np.array(xs), np.stack(ys), np.stack(vs)
        n_cells = np.prod(grid)
        eff_weight = np.array([
            var_n.sum() / y_n.sum() if y_n.sum() > 0 and var_n.sum() > 0 else 1.0 for y_n, var_n in zip(y, var_y)
        ])
        powers = x[:, None] ** np.arange(3)
        lsq = np.linalg.solve(powers.T @ (powers / eff_weight[:, None]), powers.T / eff_weight)
        coef = np.einsum("in,n...->i...", lsq, y)
        var_coef = np.einsum("in,n...->i...", lsq ** 2, var_y)
        d_mt, var_dmt = coef[1], var_coef[1]

        # nuisance effects
        exclude = re.compile(args.exclude) if args.exclude else None
        names, deltas, constraints = [], [], []
        for param, (kind, effects) in params.items():
            if param in mt_params or (exclude and exclude.search(param)):
                continue
            delta = np.zeros(grid)
            for proc, effect in effects.items():
                if kind == "lnN":
                    down, up = (
                        map(float, effect.split("/")) if "/" in effect
                        else (1.0 / float(effect), float(effect))
                    )
                    delta += 0.5 * (up - down) * nom[proc][0]
                else:
                    try:
                        delta += shape_delta(param, proc, float(effect))[0]
                    except KeyError:
                        logger.warning(f"no shapes of {param} for {proc} in {shapes}, skipped")
            names.append(param)
            deltas.append(delta)
            constraints.append(1.0)
        for proc in args.float_norm.split(",") if args.float_norm else []:
            names.append(f"norm_{proc}{tag}")
            deltas.append(nom[proc][0])
            constraints.append(0.0)

        self.names = names
        self.constraints = np.array(constraints)
        self.grid = grid
        self.dims = dims
        # significance of the linear and quadratic terms: chi2 vs. zero, about one per bin if noise only,
        # and the quality of the pol2 fit with the expected variances, chi2 / ndf about one if pol2 suffices
        ok = var_dmt > 0
        chi2_lin = np.sum(d_mt[ok] ** 2 / var_dmt[ok])
        chi2_quad = np.sum(coef[2][ok] ** 2 / var_coef[2][ok])
        # expected variances from the fitted contents, only bins with at least 10 effective events per sample
        fitted = np.einsum("ni,i...->n...", powers, coef)
        resid = y - fitted
        exp_var = eff_weight.reshape((-1,) + (1,) * len(grid)) * np.maximum(fitted, 0.0)
        fit_ok = ok & np.all(exp_var > 0, axis=0) & np.all(fitted ** 2 >= 10 * exp_var, axis=0)
        chi2_fit = np.sum(resid[:, fit_ok] ** 2 / exp_var[:, fit_ok])
        ndf = (len(x) - 3) * int(fit_ok.sum())
        logger.info(
            f"{card}: top-mass pol2 per bin, chi2 vs. zero of the slope {chi2_lin:.1f} and of the curvature "
            f"{chi2_quad:.1f} for {ok.sum()} of {n_cells} bins, fit chi2 / ndf {chi2_fit:.1f} / {ndf} "
            f"in {fit_ok.sum()} bins with at least 10 events",
        )
        v = nu + var_mc
        if np.sum((d_mt[ok] ** 2 - var_dmt[ok]) / v[ok]) <= 0:
            logger.warning(f"{card}: top-mass derivative not significant, the optimization fits noise")

        q = np.stack([nu, var_mc, sig, var_dmt, d_mt, *deltas])
        self.total = q.reshape(len(q), -1).sum(axis=1)
        # prefix sums with a leading zero along each grid axis
        p = np.zeros((len(q),) + tuple(n + 1 for n in grid))
        cum = q
        for axis in range(1, dims + 1):
            cum = np.cumsum(cum, axis=axis)
        p[(slice(None),) + (slice(1, None),) * dims] = cum
        self.prefix = p
        logger.info(
            f"{card}: category {category}, processes {procs}, {len(names)} nuisances, grid {grid}, "
            f"expected events {self.total[NU]:.6g}, signal {self.total[SIG]:.6g}",
        )

    def sums(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        # sums of all quantities over the boxes [lo, hi) (cell indices, shape (n_boxes, dims)), shape (K, n_boxes)
        out = 0.0
        for corner in itertools.product((0, 1), repeat=self.dims):
            idx = tuple(np.where(c, hi[:, k], lo[:, k]) for k, c in enumerate(corner))
            sign = (-1) ** (self.dims - sum(corner))
            out = out + sign * self.prefix[(slice(None),) + idx]
        return out

    def gram(self, s: np.ndarray) -> tuple[np.ndarray, float]:
        # fisher information terms of boxes with sums s, and the noise of the derivative of the top mass
        v = s[NU] + s[VAR_MC]
        w = s[VEC:] / np.sqrt(v)
        return w @ w.T, float(np.sum(s[VAR_DMT] / v))

    def sigma(self, g: np.ndarray, noise, params: list[int] | None = None) -> np.ndarray:
        # uncertainty on the top mass for gram matrices g (... x n x n), all nuisances or the given subset
        idx = [0] + [1 + j for j in (range(len(self.names)) if params is None else params)]
        f = np.array(g[..., idx, :][..., :, idx], dtype=np.float64)
        f[..., 0, 0] -= noise
        diag = np.arange(1, len(idx))
        f[..., diag, diag] += self.constraints[[i - 1 for i in idx[1:]]]
        f[..., np.arange(len(idx)), np.arange(len(idx))] += 1e-12 * np.abs(f[..., 0, 0])[..., None]
        e0 = np.zeros(f.shape[:-1] + (1,))
        e0[..., 0, 0] = 1.0
        try:
            var = np.linalg.solve(f, e0)[..., 0, 0]
        except np.linalg.LinAlgError:
            return np.full(f.shape[:-2], np.inf)
        return np.where(var > 0, np.sqrt(np.abs(var)), np.inf)

    def align(self, names: list[str], constraints: list[float]) -> None:
        # nuisance rows in the order of names (zero for nuisances not in this datacard), for a joint fit
        index = {name: i for i, name in enumerate(self.names)}
        p = np.zeros((VEC + 1 + len(names),) + self.prefix.shape[1:])
        p[:VEC + 1] = self.prefix[:VEC + 1]
        for j, name in enumerate(names):
            if name in index:
                p[VEC + 1 + j] = self.prefix[VEC + 1 + index[name]]
        self.prefix = p
        self.names = list(names)
        self.constraints = np.array(constraints)

    def evaluate(self, boxes: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        return evaluate([self], boxes)


def evaluate(models: list[Model], boxes: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    # uncertainty, statistical part and impacts of a joint fit of the models (aligned nuisances) with the boxes
    lo, hi = np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])
    g, noise = 0.0, 0.0
    for model in models:
        s = model.sums(lo, hi)
        if np.any(s[NU] + s[VAR_MC] <= 0):
            return {"sigma": np.inf, "stat": np.inf, "impacts": {}}
        g_m, noise_m = model.gram(s)
        g, noise = g + g_m, noise + noise_m
    ref = models[0]
    total = float(ref.sigma(g, noise))
    stat = float(ref.sigma(g, noise, []))
    impacts = {
        name: float(np.sqrt(max(float(ref.sigma(g, noise, [j])) ** 2 - stat ** 2, 0.0)))
        for j, name in enumerate(ref.names)
    }
    return {"sigma": total, "stat": stat, "impacts": impacts}


def grow_tree(models: list[Model], args: argparse.Namespace) -> list[dict]:
    # greedy splitting with the summed fisher information of the models, returns the trajectory of leaves and
    # training uncertainties
    ref = models[0]
    leaves = [(np.zeros(ref.dims, dtype=int), np.array(ref.grid, dtype=int))]
    sigma = evaluate(models, leaves)["sigma"]
    trajectory = [{"leaves": list(leaves), "sigma": sigma}]
    while len(leaves) < args.max_leaves:
        lo, hi = np.array([b[0] for b in leaves]), np.array([b[1] for b in leaves])
        s_leaves = [model.sums(lo, hi) for model in models]
        best = None
        for i, (leaf_lo, leaf_hi) in enumerate(leaves):
            # candidate children a = [lo, cut) and b = [cut, hi) along each axis
            a_lo, a_hi, b_lo, b_hi, cuts = [], [], [], [], []
            for k in range(ref.dims):
                for c in range(leaf_lo[k] + 1, leaf_hi[k]):
                    ah, bl = leaf_hi.copy(), leaf_lo.copy()
                    ah[k], bl[k] = c, c
                    a_lo.append(leaf_lo), a_hi.append(ah), b_lo.append(bl), b_hi.append(leaf_hi)
                    cuts.append((k, c))
            if not cuts:
                continue
            s_a = [model.sums(np.array(a_lo), np.array(a_hi)) for model in models]
            s_b = [model.sums(np.array(b_lo), np.array(b_hi)) for model in models]
            ok = np.ones(len(cuts), dtype=bool)
            for model, s_am, s_bm in zip(models, s_a, s_b):
                for s in (s_am, s_bm):
                    ok &= s[NU] > 0
                    ok &= s[SIG] >= args.min_signal * model.total[SIG]
                    ok &= np.sqrt(np.clip(s[VAR_MC], 0, None)) <= args.max_mc_rel * np.clip(s[NU], 1e-12, None)
            if not np.any(ok):
                continue
            cuts = [c for c, o in zip(cuts, ok) if o]
            g, noise = 0.0, 0.0
            for model, s_l, s_am, s_bm in zip(models, s_leaves, s_a, s_b):
                s_am, s_bm = s_am[:, ok], s_bm[:, ok]
                others = np.delete(s_l, i, axis=1)
                g_rest, noise_rest = model.gram(others) if others.shape[1] else (0.0, 0.0)
                v_a, v_b = s_am[NU] + s_am[VAR_MC], s_bm[NU] + s_bm[VAR_MC]
                w_a, w_b = s_am[VEC:] / np.sqrt(v_a), s_bm[VEC:] / np.sqrt(v_b)
                g = g + g_rest + np.einsum("im,jm->mij", w_a, w_a) + np.einsum("im,jm->mij", w_b, w_b)
                noise = noise + noise_rest + s_am[VAR_DMT] / v_a + s_bm[VAR_DMT] / v_b
            sig = ref.sigma(g, noise)
            m = int(np.argmin(sig))
            if best is None or sig[m] < best[0]:
                best = (float(sig[m]), i, cuts[m])
        if best is None or 1.0 / best[0] ** 2 - 1.0 / sigma ** 2 < args.min_gain / sigma ** 2:
            break
        sigma, i, (k, c) = best
        leaf_lo, leaf_hi = leaves.pop(i)
        ah, bl = leaf_hi.copy(), leaf_lo.copy()
        ah[k], bl[k] = c, c
        leaves[i:i] = [(leaf_lo, ah), (bl, leaf_hi)]
        trajectory.append({"leaves": list(leaves), "sigma": sigma, "split": (k, c)})
    return trajectory


def quantile_grid(models: list[Model], n_bins: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    # boxes of per-axis quantile bins of the signal (at fine-grid edges), axes beyond n_bins are not split
    model = models[0]
    sig = sum(m.prefix[SIG] for m in models)
    cuts = []
    for k in range(model.dims):
        if k >= len(n_bins):
            cuts.append([0, model.grid[k]])
            continue
        # cumulative signal along axis k (prefix sums at the last index of the other axes)
        cum = sig[tuple(slice(None) if j == k else -1 for j in range(model.dims))]
        cum = cum / max(cum[-1], 1e-12)
        idx = {int(np.argmin(np.abs(cum - q))) for q in np.linspace(0, 1, n_bins[k] + 1)}
        cuts.append(sorted(idx | {0, model.grid[k]}))
    return [
        (np.array([c[0] for c in cell]), np.array([c[1] for c in cell]))
        for cell in itertools.product(*[list(zip(c[:-1], c[1:])) for c in cuts])
    ]


def describe(result: dict, n_impacts: int = 5) -> str:
    top = sorted(result["impacts"].items(), key=lambda kv: -kv[1])[:n_impacts]
    return (
        f"sigma_mt={result['sigma']:.4g} GeV (stat {result['stat']:.4g}), largest impacts "
        + ", ".join(f"{name}: {value:.3g}" for name, value in top)
    )


def optimize(args: argparse.Namespace, dims: int) -> None:
    name = f"{dims}D"
    axes = binning_tree.FINE_AXES[:dims]
    edges = [binning_tree.FINE_EDGES[a] for a in axes]
    logger.info(f"========== {name}: {' x '.join(axes)} ==========")
    joint = len(args.train_card) > 1
    trains = [Model(card, args, dims, tag=f"_{i}" if joint else "") for i, card in enumerate(args.train_card)]
    if joint:
        # common nuisance list: same names are correlated between the datacards
        names = list(dict.fromkeys(name for m in trains for name in m.names))
        constraints = {name: c for m in trains for name, c in zip(m.names, m.constraints)}
        for m in trains:
            m.align(names, [constraints[name] for name in names])
    test = Model(args.test_card, args, dims) if args.test_card else None
    models = [("train", trains)] + ([("test", [test])] if test else [])

    # baselines: per-axis quantile grids
    baselines = {f"mtfit {args.grid_1d} quantile bins": [args.grid_1d]}
    if dims >= 2:
        baselines[f"{args.grid_2d} quantile grid"] = [int(n) for n in args.grid_2d.split("x")]
    if dims == 3:
        baselines[f"{args.grid_3d} quantile grid"] = [int(n) for n in args.grid_3d.split("x")]
    for label, n_bins in baselines.items():
        boxes = quantile_grid(trains, n_bins)
        for part, ms in models:
            logger.info(f"baseline {label} ({len(boxes)} bins), {part}: {describe(evaluate(ms, boxes))}")

    trajectory = grow_tree(trains, args)
    for step in trajectory:
        step["test"] = test.evaluate(step["leaves"])["sigma"] if test else np.nan
        split = ""
        if "split" in step:
            k, c = step["split"]
            split = f", split {axes[k]} at {edges[k][c]:.4g}"
        logger.info(
            f"{len(step['leaves']):3d} leaves: sigma_mt train {step['sigma']:.4g}, test {step['test']:.4g}{split}",
        )

    if args.leaves:
        chosen = min(trajectory, key=lambda st: abs(len(st["leaves"]) - args.leaves))
    elif test:
        chosen = min(trajectory, key=lambda st: st["test"])
    else:
        chosen = trajectory[-1]
    leaves = sorted(chosen["leaves"], key=lambda b: tuple(b[0]))
    logger.info(f"chosen tree with {len(leaves)} leaves")
    results = {}
    for part, ms in models:
        results[part] = evaluate(ms, leaves)
        logger.info(f"tree {name}, {part}: {describe(results[part])}")
    for i, (lo, hi) in enumerate(leaves):
        logger.info(f"  leaf {i:3d}: " + ", ".join(
            f"{a} [{e[l]:.4g}, {e[h]:.4g})" for a, e, l, h in zip(axes, edges, lo, hi)
        ))

    if args.no_write:
        return
    tree = {
        "axes": list(axes),
        "edges": edges,
        "leaves": [[[e[l], e[h]] for e, l, h in zip(edges, lo, hi)] for lo, hi in leaves],
        "sigma_mt": {part: res["sigma"] for part, res in results.items()},
        "train_card": args.train_card,
        "test_card": args.test_card,
        "settings": {
            key: getattr(args, key)
            for key in ("mt_param", "mt_delta", "exclude", "float_norm", "max_leaves", "min_signal", "max_mc_rel",
                        "min_gain", "leaves")
        },
        "trajectory": [
            {"leaves": len(st["leaves"]), "train": st["sigma"], "test": st["test"]} for st in trajectory
        ],
    }
    path = binning_tree.tree_path(name)
    if args.output_dir:
        path = os.path.join(args.output_dir, f"binning_tree_{name}.json")
    with open(path, "w") as f:
        json.dump(tree, f, indent=1, default=float)
    logger.info(f"written {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--train-card", required=True, nargs="+",
        help="fine_3D datacard(s) (.txt) used to build the tree, several for a joint fit (e.g. one per year)",
    )
    parser.add_argument("--test-card", default="", help="fine_3D datacard of another year to evaluate the tree")
    parser.add_argument("--category", default=None, help="datacard category (default: the first one)")
    parser.add_argument("--dims", default="2,3", help="dimensions of the trees, comma separated")
    parser.add_argument("--mt-param", default="mtop1,mtop3,mtop6", help="shape parameters of top-mass variations")
    parser.add_argument("--mt-delta", default="1,3,6", help="top-mass shifts of --mt-param in GeV")
    parser.add_argument("--exclude", default=r"^mtop\d", help="regex of parameters that are not nuisances")
    parser.add_argument("--float-norm", default="", help="processes with unconstrained normalization, e.g. TT,BKG")
    parser.add_argument("--max-leaves", type=int, default=40)
    parser.add_argument("--min-signal", type=float, default=0.01, help="minimum signal fraction per leaf")
    parser.add_argument("--max-mc-rel", type=float, default=0.03, help="maximum relative MC uncertainty per leaf")
    parser.add_argument("--min-gain", type=float, default=0.002, help="minimum relative gain in 1/sigma^2")
    parser.add_argument("--leaves", type=int, default=0, help="number of leaves to write (default: best test)")
    parser.add_argument("--grid-1d", type=int, default=8, help="mtfit quantile bins of the 1D baseline")
    parser.add_argument("--grid-2d", default="6x3", help="quantile grid of the 2D baseline")
    parser.add_argument("--grid-3d", default="4x3x2", help="quantile grid of the 3D baseline")
    parser.add_argument("--output-dir", default="", help="directory of the json files (default alljets/config)")
    parser.add_argument("--no-write", action="store_true", help="do not write the json files")
    args = parser.parse_args()
    for dims in map(int, args.dims.split(",")):
        optimize(args, dims)
