# coding: utf-8

"""
Mass discriminant for fully hadronic ttbar events: one network on the fitted top mass, the average reconstructed
W mass, R_bq and the b and light jet pts with one binary classification head per pair of samples, the top mass
samples 173.5 vs 171.5 GeV, the up vs down samples of the JEC (incl. bottom flavor JEC) uncertainties and the up vs
down event weights of the FSR uncertainties. The linear combination of the head scores that minimizes the expected
profile likelihood uncertainty on the top mass is the output. For a 2D fit, two further linear combinations are
produced: the mass observable with the maximal statistical sensitivity to the top mass, and the syst observable,
uncorrelated with it and without slope vs. the top mass, with the maximal information on the systematic shifts of the
mass observable (JEC and FSR), and their unrolled percentile bins.
"""

from __future__ import annotations

from typing import Any

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column
from law.target.file import get_path

from alljets.config.jet_builder import build_top1jet, build_avg_w_mass, build_reco_R_bq
from alljets.ml.spanet import SPANET_MTOP_NOMINAL, SPANET_MTOP_SHIFTS

ak = maybe_import("awkward")
np = maybe_import("numpy")
torch = maybe_import("torch")

logger = law.logger.get_logger(__name__)

# input features: log of the fitted top mass and of the average reconstructed W mass, R_bq, log of the b jet pts and
# of the light jet pts (each group sorted by decreasing pt), standardized with the nominal sample in the training
FEATURES = ["mtfit", "mwreco", "Rbq", "pt_b1", "pt_b2", "pt_q1", "pt_q2", "pt_q3", "pt_q4"]


def _build_net(hidden: list[int], n_heads: int) -> torch.nn.Module:
    # features -> shared hidden layers -> one logit per binary classification head
    layers = []
    n_in = len(FEATURES)
    for n_out in hidden:
        layers += [torch.nn.Linear(n_in, n_out), torch.nn.ReLU()]
        n_in = n_out
    layers.append(torch.nn.Linear(n_in, n_heads))
    return torch.nn.Sequential(*layers)


def _soft_memberships(s: torch.Tensor, inner_edges: torch.Tensor) -> torch.Tensor:
    # differentiable soft bin memberships (n, n_bins) of s with open outer bins, every edge is smeared by a sigmoid
    # with half the width of the narrower adjacent bin, so the memberships of each event sum to one
    widths = inner_edges[1:] - inner_edges[:-1]
    tau = 0.5 * torch.minimum(
        torch.cat([widths[:1], widths]),
        torch.cat([widths, widths[-1:]]),
    ).clamp(min=1e-6)
    above = torch.sigmoid((s.reshape(-1, 1) - inner_edges) / tau)
    ones = torch.ones_like(above[:, :1])
    above = torch.cat([ones, above, torch.zeros_like(ones)], dim=1)
    return above[:, :-1] - above[:, 1:]


def _soft_hist_cdf(s: torch.Tensor, inner_edges: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    # differentiable normalized histogram of s (with event weights w) from the soft bin memberships
    contents = _soft_memberships(s, inner_edges)
    if w is None:
        return contents.mean(dim=0)
    return (w.reshape(-1, 1) * contents).sum(dim=0) / w.sum()


def _n_eff(n: int, w=None) -> float:
    # effective number of events of a (weighted) sample of n events
    if w is None:
        return float(n)
    return float((w.sum() ** 2 / max(float((w ** 2).sum()), 1e-12)))


def _slope_coefficients(masses: list[float]) -> np.ndarray:
    # coefficients c_k of the slope of a linear fit vs. the masses, slope = sum_k c_k y_k
    m = np.asarray(masses, dtype=np.float64)
    return (m - m.mean()) / np.sum((m - m.mean()) ** 2)


# noise subtraction of the fisher matrix:
# the gram matrix G_ij = sum_b g_ib g_jb / nu_b of the template derivatives g (slope vs. the top mass and the deltas
# of the systematics) is biased by the statistical fluctuations of the templates, var(g_b) = n_events * nu_b * r for
# independent samples with the relative variance r, i.e. E[G_ii] = G_ii(true) + (n_bins - 1) * n_events * r; in a
# profiled fit, these fluctuations look like shape differences between the systematics and the top mass and
# constrain the nuisances spuriously, so the expected noise is subtracted; the deltas of event weights of the same
# (nominal) events have the per event contributions d_e, their noise is sum_e d_ie d_je / nu_b(e), incl. the
# covariance between such sources; correlations between the slope and the deltas are neglected (the nominal sample
# has no weight in the slope for symmetric mass points)


def _soft_sigma2(
    s_mass: list[torch.Tensor],
    masses: list[float],
    s_nom: torch.Tensor,
    s_syst: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor | None]]],
    n_events: float,
    n_bins: int,
) -> torch.Tensor:
    # differentiable version of _profile_sigma: profiled variance of the top mass from the noise-subtracted fisher
    # information of soft histograms in percentile bins of s_nom with open outer bins, all templates normalized to
    # n_events, the systematic samples are pairs of values and event weights (None for unweighted samples)
    q = torch.linspace(0, 1, n_bins + 1, device=s_nom.device, dtype=s_nom.dtype)[1:-1]
    # differentiable edges, the variance is invariant under monotonic linear transformations of s only if the
    # gradient includes the movement of the edges
    edges = torch.quantile(s_nom.reshape(-1), q)
    nus = torch.stack([n_events * _soft_hist_cdf(s, edges) for s in s_mass])
    nu = n_events * _soft_hist_cdf(s_nom, edges)
    # slope of a linear fit of the templates vs. the top mass in each bin
    c = _slope_coefficients(masses)
    grads = [(torch.as_tensor(c, device=s_nom.device, dtype=s_nom.dtype).reshape(-1, 1) * nus).sum(dim=0)]
    dof = n_bins - 1
    noise_diag = [dof * n_events * float(np.sum(c ** 2 / np.array([len(s) for s in s_mass])))]
    weighted = []

    def hist(sample):
        values, weights = sample
        return _soft_hist_cdf(values, edges, weights)

    for samples in s_syst.values():
        if "up" in samples and "down" in samples:
            (s_up, w_up), (s_dn, w_dn) = samples["up"], samples["down"]
            grads.append(0.5 * n_events * (hist(samples["up"]) - hist(samples["down"])))
            if w_up is not None and w_dn is not None and len(s_up) == len(s_dn):
                # event weights of the same events: per event contributions to the delta
                weighted.append((len(grads) - 1, s_up, 0.5 * n_events * (w_up / w_up.sum() - w_dn / w_dn.sum())))
                noise_diag.append(0.0)
            else:
                noise_diag.append(
                    dof * n_events * 0.25 * (1.0 / _n_eff(len(s_up), w_up) + 1.0 / _n_eff(len(s_dn), w_dn)),
                )
        else:
            sample = next(iter(samples.values()))
            grads.append(n_events * hist(sample) - nu)
            noise_diag.append(dof * n_events * (1.0 / _n_eff(len(sample[0]), sample[1]) + 1.0 / len(s_nom)))
    grads = torch.stack(grads)
    inv_nu = 1.0 / (nu + 1e-6)
    gram = grads @ (grads * inv_nu).T
    noise = torch.diag(torch.as_tensor(noise_diag, device=s_nom.device, dtype=s_nom.dtype))
    for i, s_i, d_i in weighted:
        # mean of 1 / nu over the soft bins of each event
        inv_nu_e = _soft_memberships(s_i, edges) @ inv_nu
        for j, s_j, d_j in weighted:
            if len(s_j) == len(s_i):
                noise[i, j] = (d_i * d_j * inv_nu_e).sum()
    gram = gram - noise
    # the subtraction can make the matrix indefinite, then it is shifted by its smallest eigenvalue (detached, so
    # that the gradient stays smooth)
    min_ev = torch.linalg.eigvalsh(gram.detach()).min()
    if min_ev < 0:
        gram = gram + (1e-9 - min_ev) * torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
    constraints = torch.ones(len(grads), device=s_nom.device, dtype=s_nom.dtype)
    constraints[0] = 0.0
    fisher = gram + torch.diag(constraints)
    return torch.linalg.inv(fisher)[0, 0]


def _profile_sigma(
    values: list[np.ndarray],
    masses: list[float],
    nominal_index: int,
    systs: dict[str, dict[str, tuple[np.ndarray, np.ndarray | None]]],
    n_events: float,
    n_bins: int,
    edges: np.ndarray | None = None,
) -> dict[str, Any]:
    # expected uncertainty on the top mass from a binned asimov fit in percentile bins of the nominal values (or in
    # the given bin edges), with every template normalized to n_events (shape only) and one unit-gaussian nuisance
    # per systematic, computed from the noise-subtracted fisher information F_ij = sum_b d_i nu_b d_j nu_b / nu_b
    # - noise_ij + constraints (sigma_raw without the subtraction), the systematic samples are pairs of values and
    # event weights (None for unweighted samples)
    if edges is None:
        edges = np.quantile(values[nominal_index], np.linspace(0, 1, n_bins + 1))
    edges = np.array(edges, dtype=np.float64)
    n_bins = len(edges) - 1
    edges[0], edges[-1] = -np.inf, np.inf

    def template(v, w=None):
        h = np.histogram(v, bins=edges, weights=w)[0].astype(np.float64)
        return n_events * h / max(h.sum(), 1e-12)

    def bin_index(v):
        return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)

    nus = np.stack([template(v) for v in values])
    nu = nus[nominal_index]
    c = _slope_coefficients(masses)
    d_mt = c @ nus
    keep = nu > 0
    dof = max(int(keep.sum()) - 1, 1)
    inv_nu = np.where(keep, 1.0 / np.where(keep, nu, 1.0), 0.0)
    n_nom = len(values[nominal_index])

    deltas = {}
    # relative variance of the statistical fluctuations of each delta, var(delta_b) = n_events * nu_b * noise_var
    noise_var = {}
    # per event contributions to the deltas of event weights of the same events
    weighted = {}
    for source, samples in systs.items():
        if "up" in samples and "down" in samples:
            (v_up, w_up), (v_dn, w_dn) = samples["up"], samples["down"]
            deltas[source] = 0.5 * (template(v_up, w_up) - template(v_dn, w_dn))
            if w_up is not None and w_dn is not None and len(v_up) == len(v_dn):
                weighted[source] = (v_up, 0.5 * n_events * (w_up / np.sum(w_up) - w_dn / np.sum(w_dn)))
            else:
                noise_var[source] = 0.25 * (1.0 / _n_eff(len(v_up), w_up) + 1.0 / _n_eff(len(v_dn), w_dn))
        else:
            v, w = next(iter(samples.values()))
            deltas[source] = template(v, w) - nu
            noise_var[source] = 1.0 / _n_eff(len(v), w) + 1.0 / n_nom
    # fake systematic from the two halves of the nominal sample, only used for the noise floor
    nom = values[nominal_index]
    split_delta = 0.5 * (template(nom[0::2]) - template(nom[1::2]))

    names = ["mt", *deltas, "nominal_split"]
    vecs = np.stack([d_mt, *deltas.values(), split_delta])
    gram = vecs @ (vecs * inv_nu).T
    noise = np.zeros_like(gram)
    noise[0, 0] = dof * n_events * float(np.sum(c ** 2 / np.array([len(v) for v in values])))
    noise[-1, -1] = dof * n_events / n_nom
    for source, var in noise_var.items():
        noise[names.index(source), names.index(source)] = dof * n_events * var
    for a, (v_a, d_a) in weighted.items():
        inv_nu_e = inv_nu[bin_index(v_a)]
        for b, (v_b, d_b) in weighted.items():
            if len(v_b) == len(v_a):
                noise[names.index(a), names.index(b)] = float(np.sum(d_a * d_b * inv_nu_e))
        noise_var[a] = noise[names.index(a), names.index(a)] / (dof * n_events)
    gram_sub = gram - noise

    def sigma(sources, subtract=True):
        idx = [0] + [names.index(s) for s in sources]
        g = (gram_sub if subtract else gram)[np.ix_(idx, idx)]
        # the subtraction can make the matrix indefinite, negative eigenvalues are clamped to zero
        ev, evec = np.linalg.eigh(g)
        g = (evec * np.clip(ev, 0.0, None)) @ evec.T
        fisher = g + np.diag([0.0] + [1.0] * len(sources))
        fisher[0, 0] = max(fisher[0, 0], 1e-12)
        return float(np.sqrt(max(np.linalg.inv(fisher)[0, 0], 0.0)))

    # prefit impacts: shift of the fitted top mass for a +1 sigma variation without constraint of the
    # nuisance by the data, i.e. sum_b d_b delta_b / nu_b / sum_b d_b^2 / nu_b (noise-subtracted denominator)
    fisher_mt = max(gram_sub[0, 0], 1e-12)
    prefit = {s: float(gram[0, names.index(s)] / fisher_mt) for s in deltas}

    sigma_stat = sigma([])
    # noise floor: the statistical fluctuations of the templates alone give prefit impacts with an rms of
    # sigma_stat * sqrt(n_events * noise_var), checked by the fake systematic of the split nominal sample
    split_prefit = float(gram[0, -1] / fisher_mt)
    return {
        "sigma": sigma(list(deltas)),
        "sigma_raw": sigma(list(deltas), subtract=False),
        "sigma_stat": sigma_stat,
        "impacts": {s: float(np.sqrt(max(sigma([s]) ** 2 - sigma_stat ** 2, 0.0))) for s in deltas},
        "prefit_impacts": prefit,
        "sigma_prefit": float(np.sqrt(sigma_stat ** 2 + sum(v ** 2 for v in prefit.values()))),
        "noise": {s: float(sigma_stat * np.sqrt(n_events * var)) for s, var in noise_var.items()},
        "split_impact": float(np.sqrt(max(sigma(["nominal_split"]) ** 2 - sigma_stat ** 2, 0.0))),
        "split_prefit": split_prefit,
        "split_noise": float(sigma_stat * np.sqrt(n_events / n_nom)),
    }


def _cdf_transform(s: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    # piecewise linear map of s to its cumulative fraction given by equidistant quantiles
    levels = torch.linspace(0.0, 1.0, len(quantiles), dtype=s.dtype, device=s.device)
    i = torch.searchsorted(quantiles, s.contiguous()).clamp(1, len(quantiles) - 1)
    lo, hi = quantiles[i - 1], quantiles[i]
    t = ((s - lo) / (hi - lo).clamp(min=1e-12)).clamp(0.0, 1.0)
    return levels[i - 1] + t * (levels[i] - levels[i - 1])


class _Combination:
    # linear combination of the standardized classifier logits, mapped to its cumulative fraction in the nominal
    # sample so that the output is uniform in [0, 1], and the two observables of the 2D fit (mass and syst), two
    # further linear combinations (columns of weights_2d) mapped to their cumulative fractions (rows of quantiles_2d)
    # (no torch.nn.Module subclass, torch is only a mock module outside of the sandbox)
    def __init__(self, net, t_mean, t_std, weights, quantiles, weights_2d=None, quantiles_2d=None):
        self.net = net
        self.t_mean, self.t_std, self.weights, self.quantiles = t_mean, t_std, weights, quantiles
        self.weights_2d, self.quantiles_2d = weights_2d, quantiles_2d

    def eval(self):
        self.net.eval()
        return self

    def logits(self, X):
        # standardized logits of all classification heads, shape (n, n_heads)
        return (self.net(X) - self.t_mean.to(X)) / self.t_std.to(X)

    def __call__(self, X):
        t = self.logits(X) @ self.weights.to(X)
        return _cdf_transform(t, self.quantiles.to(X)).reshape(-1, 1)

    def observables_2d(self, X):
        # mass and syst observables of the 2D fit, shape (n, 2), each uniform in [0, 1] for nominal events
        t = self.logits(X) @ self.weights_2d.to(X)
        return torch.stack([
            _cdf_transform(t[:, i].contiguous(), self.quantiles_2d[i].to(X)) for i in range(t.shape[1])
        ], dim=1)


class MassClassifierDNN(MLModel):

    # mark the model as accepting only a single config
    single_config = True

    # defaults of the settings, overridden by --ml-model-settings or by cls_dict in derivations
    # (lists are given colon separated, e.g. jec_sources=AbsoluteMPFBias:FlavorPureBottom)
    jec_sources = "AbsoluteMPFBias:RelativeSample:FlavorPureBottom"
    # additional up/down shifts with separate samples (e.g. tune_rtt), none by default
    additional_shifts = ""
    # up/down shifts given by event weights of the nominal events (e.g. parton shower weights)
    weight_shifts = "fsr_Q2QG_muR:fsr_X2XG_muR"
    hidden = "32:32"
    # expected number of events and bins of the profile likelihood estimate of the top mass uncertainty
    lik_events = 30000
    lik_bins = 10
    # only mass points within this distance (GeV) to the nominal top mass are used for the slope of the templates
    slope_window = 3.0
    epochs = 100
    batch_size = 65536
    lr = 1e-3
    # decoupled weight decay of AdamW of the whole network against overfitting statistical fluctuations
    weight_decay = 1e-4
    # early stopping after patience epochs without improvement of the sum of the validation losses of all heads
    patience = 8
    # learning rate reduction by lr_factor after lr_patience epochs without improvement of the validation loss
    lr_factor = 0.5
    lr_patience = 3
    # fraction of the events not used in the training of the classifiers, one half is used for the early stopping
    # and the fit of the combination weights, the other half for the evaluation
    val_fraction = 0.4
    # optimization steps from the linear-moment optimum, initial and final learning rate (cosine decay) of the fit
    # of the combination weights, and the interval of the evaluation of the hard profile likelihood estimate
    # (0: linear-moment optimum only, deterministic; the refinement gained less than the noise of the estimate)
    comb_steps = 0
    comb_lr = 0.01
    comb_lr_min = 1e-3
    comb_eval_every = 25
    # folds of the pooled comb and test parts for the fit and evaluation of the combination weights
    comb_folds = 4
    # fraction of the steps at the end over which the hard estimate is averaged for the comparison with the start
    comb_tail_fraction = 0.25
    # bootstrap resamples for the uncertainty of the profile likelihood estimates on the test part (0: none)
    n_boot = 50
    # only events in this category are used in the training
    train_category = "sig"
    # events with a fitted top mass above mtfit_max (GeV) get the output tail_output in a separate bin
    mtfit_max = 300.0
    tail_output = 1.05
    # percentile bins of the mass and of the syst observable of the 2D fit (unrolled, mass bins inner), used for the
    # unrolled output and the profile likelihood estimate of the 2D fit
    bins_2d = "10:4"
    # bin counts of the scan of the held-out profile likelihood estimates with the k-fold weights (1D: mtfit and the
    # combination, 2D: mass x syst and mtfit x syst, given as n_mass x n_syst), empty: no scan
    scan_bins = "5:8:10:15:20"
    scan_bins_2d = "10x2:10x3:10x4:10x6:15x4"

    def __init__(self, *args, folds: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, the attributes refer to class-level attributes)
        get = self.parameters.get
        self.folds = int(get("folds", folds or self.folds))
        self.jec_sources = str(get("jec_sources", self.jec_sources)).split(":")
        self.additional_shifts = [s for s in str(get("additional_shifts", self.additional_shifts)).split(":") if s]
        self.weight_shifts = [s for s in str(get("weight_shifts", self.weight_shifts)).split(":") if s]
        self.hidden = [int(n) for n in str(get("hidden", self.hidden)).split(":")]
        self.lik_events = float(get("lik_events", self.lik_events))
        self.lik_bins = int(get("lik_bins", self.lik_bins))
        self.slope_window = float(get("slope_window", self.slope_window))
        self.epochs = int(get("epochs", self.epochs))
        self.batch_size = int(get("batch_size", self.batch_size))
        self.lr = float(get("lr", self.lr))
        self.weight_decay = float(get("weight_decay", self.weight_decay))
        self.patience = int(get("patience", self.patience))
        self.lr_factor = float(get("lr_factor", self.lr_factor))
        self.lr_patience = int(get("lr_patience", self.lr_patience))
        self.val_fraction = float(get("val_fraction", self.val_fraction))
        self.comb_steps = int(get("comb_steps", self.comb_steps))
        self.comb_lr = float(get("comb_lr", self.comb_lr))
        self.comb_lr_min = float(get("comb_lr_min", self.comb_lr_min))
        self.comb_eval_every = int(get("comb_eval_every", self.comb_eval_every))
        self.comb_folds = int(get("comb_folds", self.comb_folds))
        self.comb_tail_fraction = float(get("comb_tail_fraction", self.comb_tail_fraction))
        self.n_boot = int(get("n_boot", self.n_boot))
        self.train_category = str(get("train_category", self.train_category))
        self.mtfit_max = float(get("mtfit_max", self.mtfit_max))
        self.tail_output = float(get("tail_output", self.tail_output))
        self.bins_2d = [int(n) for n in str(get("bins_2d", self.bins_2d)).split(":")]
        if len(self.bins_2d) != 2:
            raise ValueError(f"bins_2d needs two numbers of bins (mass:syst), got {self.bins_2d}")
        self.scan_bins = [int(n) for n in str(get("scan_bins", self.scan_bins)).split(":") if n]
        self.scan_bins_2d = [
            tuple(int(n) for n in b.split("x")) for b in str(get("scan_bins_2d", self.scan_bins_2d)).split(":") if b
        ]

        # mass classes sorted by the generated top mass
        self.mass_points = sorted([SPANET_MTOP_NOMINAL, *SPANET_MTOP_SHIFTS.values()])

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        if f"{self.cls_name}_output" not in self.config_inst.variables:
            self.config_inst.add_variable(
                name=f"{self.cls_name}_output",
                expression=f"{self.cls_name}.output",
                null_value=-1,
                # bin width 0.025, the last 4 bins hold the events with mtfit > mtfit_max
                binning=(44, 0.0, 1.1),
                x_title=f"{self.cls_name} output",
            )
        if f"{self.cls_name}_output_percentile" not in self.config_inst.variables:
            self.config_inst.add_variable(
                name=f"{self.cls_name}_output_percentile",
                expression=f"{self.cls_name}.output",
                null_value=-1,
                # the output is uniform for nominal events by construction, so equal bins are percentile bins,
                # the 11th bin holds the events with mtfit > mtfit_max (output tail_output)
                binning=(11, 0.0, 1.1),
                x_title=f"{self.cls_name} output",
            )
        # observables of the 2D fit, uniform for nominal events, tail events at tail_output
        for obs in ("mass", "syst"):
            if f"{self.cls_name}_{obs}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}_{obs}",
                    expression=f"{self.cls_name}.{obs}",
                    null_value=-1,
                    binning=(44, 0.0, 1.1),
                    x_title=f"{self.cls_name} {obs} observable",
                )
        # unrolled percentile bins of the 2D fit, bin = i_mass + n_mass * i_syst, the last bin holds the events with
        # mtfit > mtfit_max
        n_2d = self.bins_2d[0] * self.bins_2d[1]
        if f"{self.cls_name}_unrolled" not in self.config_inst.variables:
            self.config_inst.add_variable(
                name=f"{self.cls_name}_unrolled",
                expression=f"{self.cls_name}.unrolled",
                null_value=-1,
                binning=(n_2d + 1, -0.5, n_2d + 0.5),
                x_title=f"{self.cls_name} unrolled bin (mass x syst)",
            )

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$AJ_BASE/sandboxes/spanet.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {config_inst.get_dataset("tt_fh_powheg")}

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return (
            {f"{coll}.{f}" for coll in ("FitTop1", "RecoW1", "RecoW2") for f in "xyzt"} |
            {f"FitJet.reco.{f}" for f in ("pt", "eta", "phi", "mass")} |
            {"FitChi2", "category_ids"} |
            set(self.weight_columns(config_inst).values())
        )

    def weight_columns(self, config_inst: od.Config) -> dict[str, str]:
        # normalized event weight columns of the weight shifts, taken from the shift aliases, e.g. fsr_Q2QG_muR_up ->
        # normalized_fsr_weight_Q2QG_muR_up (produced by normalized_ps_weights for the nominal shift)
        columns = {}
        for source in self.weight_shifts:
            for direction in ("up", "down"):
                shift = f"{source}_{direction}"
                if not config_inst.has_shift(shift):
                    continue
                aliases = config_inst.get_shift(shift).get_aux("column_aliases", {})
                normalized = [col for name, col in aliases.items() if name.startswith("normalized_")]
                if len(normalized) != 1:
                    raise ValueError(
                        f"no unique normalized event weight column in the aliases of shift {shift}: {aliases}",
                    )
                columns[shift] = normalized[0]
        return columns

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        return {f"{self.cls_name}.{col}" for col in ("output", "mass", "syst", "unrolled")}

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

    def syst_shifts(self, config_inst: od.Config) -> list[str]:
        # up/down shifts the output should be insensitive to
        sources = self.additional_shifts + [f"jec_{source}" for source in self.jec_sources]
        return [
            f"{source}_{direction}"
            for source in sources
            for direction in ("up", "down")
            if config_inst.has_shift(f"{source}_{direction}")
        ]

    def training_shifts(self, config_inst: od.Config, dataset_inst: od.Dataset) -> set[str]:
        # shifts with separate samples (top mass variations, top recoil) used in the training
        return {
            shift for shift in list(SPANET_MTOP_SHIFTS) + self.syst_shifts(config_inst)
            if config_inst.has_shift(shift) and shift in dataset_inst.info
        }

    def requires(self, task: law.Task) -> dict[str, dict[str, law.Task]]:
        # PrepareMLEvents workflows of the top mass variations and of the systematic shifts,
        # MLTraining only requires the nominal events
        from columnflow.tasks.ml import PrepareMLEvents

        reqs = {}
        for dataset_inst in self.datasets(self.config_inst):
            shifts = {
                shift for shift in SPANET_MTOP_SHIFTS
                if shift in self.training_shifts(self.config_inst, dataset_inst)
            } | set(self.syst_shifts(self.config_inst))
            reqs[dataset_inst.name] = {
                shift: PrepareMLEvents.req(
                    task,
                    config=self.config_inst.name,
                    dataset=dataset_inst.name,
                    shift=shift,
                    branch=-1,
                    _exclude={"branches"},
                )
                for shift in sorted(shifts)
            }
        return reqs

    def _features(self, events: ak.Array) -> tuple[np.ndarray, np.ndarray]:
        # feature matrix (n, len(FEATURES)) and mask of events with a converged fit and finite features,
        # the features are only built for fitted events
        fitted = ak.to_numpy(ak.fill_none(events.FitChi2, -1.0)) >= 0
        X = np.full((len(events), len(FEATURES)), np.nan, dtype=np.float32)
        if np.any(fitted):
            fitted_events = events[fitted]
            masses = np.stack([
                ak.to_numpy(ak.fill_none(col, np.nan))
                for col in (build_top1jet(fitted_events, which="mass"), build_avg_w_mass(fitted_events))
            ], axis=1)
            r_bq = ak.to_numpy(ak.fill_none(build_reco_R_bq(fitted_events), np.nan)).reshape(-1, 1)
            # pts of the fitted jets, 0-1 are the b jets and 2-5 the light jets, sorted by decreasing pt per group
            pt = ak.to_numpy(ak.fill_none(ak.pad_none(fitted_events.FitJet.reco.pt, 6, clip=True), np.nan))
            pt = np.concatenate([-np.sort(-pt[:, :2], axis=1), -np.sort(-pt[:, 2:6], axis=1)], axis=1)
            # log of masses and pts, non-positive values give non-finite features and are masked below
            with np.errstate(divide="ignore", invalid="ignore"):
                X[fitted] = np.concatenate([np.log(masses), r_bq, np.log(pt)], axis=1)
        valid = fitted & np.all(np.isfinite(X), axis=1)
        return X, valid

    def _load_features(
        self,
        targets: list[law.FileSystemFileTarget],
        weight_columns: list[str] = (),
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # features and event weights of the valid events in the training category
        category_id = self.config_inst.get_category(self.train_category).id
        arrays, weights = [], {col: [] for col in weight_columns}
        for target in targets:
            events = ak.from_parquet(get_path(target))
            X, valid = self._features(events)
            in_category = ak.to_numpy(ak.any(events.category_ids == category_id, axis=1))
            mask = valid & in_category
            arrays.append(X[mask])
            for col in weight_columns:
                weights[col].append(ak.to_numpy(events[col]).astype(np.float32)[mask])
        return np.concatenate(arrays, axis=0), {col: np.concatenate(w) for col, w in weights.items()}

    def _training_samples(self, task: law.Task, input: dict[str, Any]) -> tuple[list[np.ndarray], dict]:
        # features of the mass points (sorted by mass, nominal included) and pairs of features and event weights of
        # the systematic up/down shifts, the weight shifts have no features (the nominal events) and only weights
        targets = {SPANET_MTOP_NOMINAL: [
            inp["mlevents"]
            for dataset_files in input["events"][self.config_inst.name].values()
            for inp in dataset_files
        ]}
        syst_targets = {}
        for shift_inputs in input["model"].values():
            for shift, inp in shift_inputs.items():
                files = [
                    branch_output["mlevents"][f]
                    for branch_output in inp["collection"].targets.values()
                    for f in task.training_folds
                ]
                if shift in SPANET_MTOP_SHIFTS:
                    targets.setdefault(SPANET_MTOP_SHIFTS[shift], []).extend(files)
                else:
                    syst_targets.setdefault(shift, []).extend(files)

        missing = set(self.mass_points) - set(targets)
        if missing:
            raise ValueError(f"no training events for top masses {sorted(missing)}")

        weight_columns = self.weight_columns(self.config_inst)
        mass_samples = []
        for mass in self.mass_points:
            if mass == SPANET_MTOP_NOMINAL:
                X, nominal_weights = self._load_features(targets[mass], list(weight_columns.values()))
            else:
                X, _ = self._load_features(targets[mass])
            mass_samples.append(X)
        syst_samples = {}
        for shift in sorted(syst_targets):
            source, direction = shift.rsplit("_", 1)
            syst_samples.setdefault(source, {})[direction] = (self._load_features(syst_targets[shift])[0], None)
        for shift, col in sorted(weight_columns.items()):
            source, direction = shift.rsplit("_", 1)
            syst_samples.setdefault(source, {})[direction] = (None, nominal_weights[col])
        return mass_samples, syst_samples

    def train(
            self,
            task: law.Task,
            input: dict[str, Any],
            output: law.FileSystemDirectoryTarget,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(task.branch)
        # reproducible initialization and batches of the network
        torch.manual_seed(task.branch)

        mass_samples, syst_samples = self._training_samples(task, input)
        nominal_class = self.mass_points.index(SPANET_MTOP_NOMINAL)
        for mass, X in zip(self.mass_points, mass_samples):
            logger.info(f"mtop {mass}: {len(X)} events")
        for source, samples in syst_samples.items():
            logger.info(f"{source}: " + ", ".join(
                f"{d} {len(X)} events" if X is not None else f"{d} nominal events with mean weight {w.mean():.4g}"
                for d, (X, w) in samples.items()
            ))

        # standardization with the nominal events
        mean = mass_samples[nominal_class].mean(axis=0)
        std = mass_samples[nominal_class].std(axis=0)

        # every sample is split into the training part of the classifiers, the comb part for their early stopping
        # and the fit of the combination weights, and the test part for the evaluation
        def slices(n):
            n_val = int(n * self.val_fraction)
            n_comb = n_val // 2
            return {"comb": slice(0, n_comb), "test": slice(n_comb, n_val), "train": slice(n_val, None)}

        def split(X, perm):
            X = ((X - mean) / std)[perm]
            return {part: torch.as_tensor(X[sl], device=device) for part, sl in slices(len(X)).items()}

        perms = [rng.permutation(len(X)) for X in mass_samples]
        mass_parts = [split(X, perm) for X, perm in zip(mass_samples, perms)]

        def split_syst(X, w):
            # parts of pairs of features and event weights, the weight shifts use the nominal events with the same
            # split as the nominal sample
            if w is None:
                return {part: (x, None) for part, x in split(X, rng.permutation(len(X))).items()}
            w = w[perms[nominal_class]]
            return {
                part: (mass_parts[nominal_class][part], torch.as_tensor(w[sl], device=device))
                for part, sl in slices(len(w)).items()
            }

        syst_parts = {
            src: {d: split_syst(X, w) for d, (X, w) in samples.items()}
            for src, samples in syst_samples.items()
        }

        # mass points close to the nominal mass used for the slope of the templates vs. the top mass
        slope_points = [
            i for i, m in enumerate(self.mass_points) if abs(m - SPANET_MTOP_NOMINAL) <= self.slope_window
        ]
        slope_masses = [self.mass_points[i] for i in slope_points]
        slope_nominal = slope_points.index(nominal_class)
        logger.info(f"mass points of the slope vs. the top mass: {slope_masses}")

        # separate generator for the bootstrap, so that it does not change the splits
        boot_rng = np.random.default_rng(task.branch + 1000)

        def profile(mass_X, syst_X, f, n_boot=0, edges=None, n_bins=None):
            # profile likelihood estimate for the values f(X) of the given samples, the weight shifts use the
            # values of the nominal events; with n_boot > 0, the mean and rms of sigma over bootstrap resamples of
            # all samples are added, the weighted samples are resampled with the indices of the nominal sample;
            # n_bins (default lik_bins) percentile bins of the nominal values, or the given bin edges
            n_bins = n_bins or self.lik_bins
            mass_v = [f(mass_X[i]) for i in slope_points]
            nom_v = mass_v[slope_nominal]
            syst_v = {
                src: {
                    d: (f(X), None) if w is None else (nom_v, w.cpu().numpy())
                    for d, (X, w) in samples.items()
                }
                for src, samples in syst_X.items()
            }
            lik = _profile_sigma(
                mass_v, slope_masses, slope_nominal, syst_v, self.lik_events, n_bins, edges,
            )
            if n_boot > 0:
                sigmas = []
                for _ in range(n_boot):
                    idx = [boot_rng.integers(len(v), size=len(v)) for v in mass_v]
                    nom_idx = idx[slope_nominal]
                    boot_syst = {
                        src: {
                            d: (v[boot_rng.integers(len(v), size=len(v))], None) if w is None
                            else (v[nom_idx], w[nom_idx])
                            for d, (v, w) in samples.items()
                        }
                        for src, samples in syst_v.items()
                    }
                    sigmas.append(_profile_sigma(
                        [v[i] for v, i in zip(mass_v, idx)], slope_masses, slope_nominal, boot_syst,
                        self.lik_events, n_bins, edges,
                    )["sigma"])
                lik["sigma_boot_mean"] = float(np.mean(sigmas))
                lik["sigma_boot_rms"] = float(np.std(sigmas))
            return lik

        def profile_2d(mass_X, syst_X, f, n_boot=0, bins=None):
            # profile likelihood estimate of a 2D fit of the two values f(X) (n, 2) in the unrolled bins of their
            # percentile bins (bins_2d) for the nominal events, mass bins inner
            n_a, n_b = bins or self.bins_2d
            nom = f(mass_X[nominal_class])
            inner = [np.quantile(nom[:, i], np.linspace(0, 1, n + 1)[1:-1]) for i, n in enumerate((n_a, n_b))]

            def unrolled(X):
                v = f(X)
                i_a, i_b = (np.searchsorted(e, v[:, i], side="right") for i, e in enumerate(inner))
                return (i_a + n_a * i_b).astype(np.float64)

            return profile(mass_X, syst_X, unrolled, n_boot, edges=np.arange(n_a * n_b + 1) - 0.5)

        def fmt_lik(lik):
            boot = (
                f"bootstrap {lik['sigma_boot_mean']:.4g} +- {lik['sigma_boot_rms']:.2g}, "
                if "sigma_boot_rms" in lik else ""
            )
            return (
                f"sigma_mt={lik['sigma']:.4g} GeV ({boot}raw {lik['sigma_raw']:.4g}, stat {lik['sigma_stat']:.4g}, "
                f"prefit {lik['sigma_prefit']:.4g}), "
                f"impacts (prefit, noise) " +
                " ".join(
                    f"{s}: {v:.3g} ({lik['prefit_impacts'][s]:+.3g}, {lik['noise'][s]:.2g})"
                    for s, v in lik["impacts"].items()
                ) +
                f", nominal split: {lik['split_impact']:.3g} ({lik['split_prefit']:+.3g}, {lik['split_noise']:.2g})"
            )

        # pairs of samples (label 1, label 0) of the classifiers: top mass +-1 GeV and up vs down of each source
        # (dicts of parts of pairs of features and event weights)
        pairs = {
            "mtop": tuple(
                {part: (X, None) for part, X in mass_parts[self.mass_points.index(SPANET_MTOP_NOMINAL + d)].items()}
                for d in (1.0, -1.0)
            ),
        }
        for source, samples in syst_parts.items():
            if "up" in samples and "down" in samples:
                pairs[source] = (samples["up"], samples["down"])
            else:
                logger.warning(f"{source}: no up and down samples, no classifier")

        names = list(pairs)

        def bce(logits1, w1, logits0, w0):
            # binary cross entropy with event weights and equal total weights of both classes
            n = len(logits1) + len(logits0)

            def class_weights(logits, w):
                w = torch.ones_like(logits) if w is None else w.to(logits)
                return w * (0.5 * n / w.sum())

            logits = torch.cat([logits1, logits0])
            labels = torch.cat([torch.ones_like(logits1), torch.zeros_like(logits0)])
            weights = torch.cat([class_weights(logits1, w1), class_weights(logits0, w0)])
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)

        def head_losses(part, n=None):
            # loss of each head on its pair of samples, balanced batches of n events per class in the training
            losses = {}
            for k, name in enumerate(names):
                (X1, w1), (X0, w0) = (p[part] if n is None else sample(p[part], n) for p in pairs[name])
                losses[name] = bce(net(X1)[:, k], w1, net(X0)[:, k], w0)
            return losses

        def sample(pair, n):
            X, w = pair
            idx = torch.randint(len(X), (n,), device=device)
            return X[idx], None if w is None else w[idx]

        # one network with shared hidden layers and one head per classifier, trained on the sum of the head losses
        net = _build_net(self.hidden, len(names)).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience,
            # the improvements of the loss are O(1e-5) per epoch, the default relative threshold of 1e-4 would
            # reduce the learning rate while the network is still learning
            threshold_mode="abs", threshold=1e-6,
        )
        half = max(self.batch_size // 2, 1)
        steps = max(
            max(len(p1["train"][0]) + len(p0["train"][0]) for p1, p0 in pairs.values()) // self.batch_size, 1,
        )
        clf_history = []
        best_loss, best_state, bad_epochs = np.inf, None, 0
        for epoch in range(self.epochs):
            net.train()
            train_losses = dict.fromkeys(names, 0.0)
            for _ in range(steps):
                losses = head_losses("train", n=half)
                for name, loss in losses.items():
                    train_losses[name] += loss.item() / steps
                optimizer.zero_grad()
                sum(losses.values()).backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_losses = {name: loss.item() for name, loss in head_losses("comb").items()}
            val_loss = sum(val_losses.values())
            record = {
                "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "val_bce_sum": val_loss,
                **{f"train_bce_{name}": v for name, v in train_losses.items()},
                **{f"val_bce_{name}": v for name, v in val_losses.items()},
            }
            clf_history.append(record)
            logger.info(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    logger.info(f"early stopping after epoch {epoch}")
                    break

        logger.info(f"best validation bce sum {best_loss:.5g}")
        net.load_state_dict(best_state)
        net.eval()

        # the comb and test parts are pooled for the k-fold fit of the combination weights: the weights are fitted
        # on k-1 folds and evaluated on the held-out fold, the mean over the folds is an unbiased estimate with the
        # statistics of both parts; the folds of the weight shifts are the folds of the nominal sample
        def pool(parts):
            return torch.cat([parts["comb"], parts["test"]])

        mass_pool = [pool(p) for p in mass_parts]
        syst_pool = {
            src: {
                d: (
                    torch.cat([p["comb"][0], p["test"][0]]),
                    None if p["comb"][1] is None else torch.cat([p["comb"][1], p["test"][1]]),
                )
                for d, p in samples.items()
            }
            for src, samples in syst_parts.items()
        }
        n_folds = max(self.comb_folds, 2)

        def take(mass_X, syst_X, k, held_out):
            # fold k (held_out) or all other folds of every sample, contiguous folds of the shuffled samples
            def sel(X):
                in_fold = torch.arange(len(X), device=X.device) * n_folds // len(X) == k
                return X[in_fold] if held_out else X[~in_fold]
            return (
                [sel(X) for X in mass_X],
                {
                    src: {d: (sel(X), None if w is None else sel(w)) for d, (X, w) in samples.items()}
                    for src, samples in syst_X.items()
                },
            )

        # standardized logits of all heads, standardized with the nominal training events
        comb = _Combination(net, torch.zeros(len(names)), torch.ones(len(names)), torch.zeros(len(names)), None)
        with torch.no_grad():
            T_nom = comb.logits(mass_parts[nominal_class]["train"])
            comb.t_mean, comb.t_std = T_nom.mean(dim=0).cpu(), T_nom.std(dim=0).clamp(min=1e-6).cpu()
            T_mass_pool = [comb.logits(X) for X in mass_pool]
            T_syst_pool = {
                src: {d: (comb.logits(X), w) for d, (X, w) in samples.items()}
                for src, samples in syst_pool.items()
            }

        def linear(w):
            # values of the linear combination w of the standardized logits
            return lambda T: (T @ w).cpu().numpy()

        def moments(T_mass, T_syst):
            # linear moments of the standardized logits T: slope mu_mt of the mean of T vs. the top mass, covariance
            # C of T for nominal events and S = sum_j mu_j mu_j^T of the shifts mu_j of the mean of T by +1 sigma of
            # source j; the statistical noise of the estimated mu_j is subtracted (E[mu mu^T] = mu mu^T + cov(mu))
            # and S is clamped to be positive
            with torch.no_grad():
                f64 = torch.float64
                c = torch.as_tensor(_slope_coefficients(slope_masses), dtype=f64, device=device)
                mu_mt = (c.reshape(-1, 1) * torch.stack([T_mass[i].to(f64).mean(dim=0) for i in slope_points])).sum(0)
                T_nom = T_mass[nominal_class].to(f64)
                cov = torch.cov(T_nom.T)

                def mean_and_cov(T, w):
                    # weighted mean of T and the covariance of this mean
                    T = T.to(f64)
                    if w is None:
                        return T.mean(dim=0), torch.cov(T.T) / len(T)
                    w = w.to(f64)
                    m = (w.reshape(-1, 1) * T).sum(dim=0) / w.sum()
                    return m, torch.cov(T.T) * (w ** 2).sum() / w.sum() ** 2

                syst = torch.zeros_like(cov)
                for samples in T_syst.values():
                    if "up" in samples and "down" in samples:
                        (T_up, w_up), (T_dn, w_dn) = samples["up"], samples["down"]
                        if w_up is not None and w_dn is not None and len(T_up) == len(T_dn):
                            # event weights of the same events: mu = sum_e a_e T_e with per event coefficients a_e
                            a = 0.5 * (w_up.to(f64) / w_up.sum() - w_dn.to(f64) / w_dn.sum())
                            T = T_up.to(f64)
                            mu = (a.reshape(-1, 1) * T).sum(dim=0)
                            T_c = T - T.mean(dim=0)
                            noise = (a.reshape(-1, 1) ** 2 * T_c).T @ T_c
                        else:
                            (m_up, v_up), (m_dn, v_dn) = mean_and_cov(T_up, w_up), mean_and_cov(T_dn, w_dn)
                            mu = 0.5 * (m_up - m_dn)
                            noise = 0.25 * (v_up + v_dn)
                    else:
                        m, v = mean_and_cov(*next(iter(samples.values())))
                        mu = m - T_nom.mean(dim=0)
                        noise = v + cov / len(T_nom)
                    syst = syst + torch.outer(mu, mu) - noise
                ev, evec = torch.linalg.eigh(syst)
                return mu_mt, cov, (evec * ev.clamp(min=0.0)) @ evec.T

        def linear_weights(T_mass, T_syst):
            # global optimum of the linear-moment approximation: with the mean of s = T w as the only observable,
            # sigma_mt^2 = (w^T C w / N + sum_j (w . mu_j)^2) / (w . mu_mt)^2; this generalized rayleigh quotient is
            # minimized by w = A^-1 mu_mt with A = C / N + S, no local minima
            with torch.no_grad():
                mu_mt, cov, syst = moments(T_mass, T_syst)
                A = cov / self.lik_events + syst
                w = torch.linalg.solve(A, mu_mt)
                sigma_lin = float(torch.sqrt(w @ A @ w) / (w @ mu_mt).abs())
                return (w / w.norm()).to(T_mass[0].dtype), sigma_lin

        def weights_2d(T_mass, T_syst):
            # weights (n_heads, 2) of the two observables of the 2D fit:
            # mass: w_a = C^-1 mu_mt, maximal statistical sensitivity to the top mass (fisher discriminant);
            # syst: v maximizes the information on the systematic shift of the mass observable, a_j = w_a . mu_j,
            # i.e. (v^T S w_a)^2 / v^T A v with A = C / N + S, under the constraint v . mu_mt = 0: no slope vs. the
            # top mass and, as C w_a = mu_mt, uncorrelated with the mass observable for nominal events; solution
            # v = A^-1 (S w_a - lambda mu_mt); span(w_a, v) contains the 1D optimum A^-1 mu_mt, so the mean-only
            # estimate of the 2D fit equals the one of the 1D combination, the 2D fit adds the shape information;
            # returns the weights, the mean-only estimate sigma^2 = 1 / (m^T (W^T A W)^-1 m) with m = W^T mu_mt,
            # and the correlation of the two observables for nominal events
            with torch.no_grad():
                mu_mt, cov, syst = moments(T_mass, T_syst)
                A = cov / self.lik_events + syst
                w_a = torch.linalg.solve(cov, mu_mt)
                a_g = torch.linalg.solve(A, syst @ w_a)
                a_mu = torch.linalg.solve(A, mu_mt)
                v = a_g - (mu_mt @ a_g) / (mu_mt @ a_mu) * a_mu
                W = torch.stack([w_a / w_a.norm(), v / v.norm().clamp(min=1e-12)], dim=1)
                m = W.T @ mu_mt
                sigma_lin = float(torch.rsqrt(m @ torch.linalg.solve(W.T @ A @ W, m)))
                c2 = W.T @ cov @ W
                corr = float(c2[0, 1] / torch.sqrt(c2[0, 0] * c2[1, 1]))
                return W.to(T_mass[0].dtype), sigma_lin, corr

        def fit_weights(T_mass, T_syst, prefix):
            # fit of the combination weights: the linear-moment optimum (global, no local minima) is the start, the
            # soft profiled variance of the top mass with binned templates of the linear combination refines it with
            # the shape information; only the direction of the weights matters, so they are normalized to unit
            # length, the learning rate is reduced from comb_lr to comb_lr_min (cosine); the refined weights (mean
            # of the hard profile likelihood estimate over the converged tail, last comb_tail_fraction of the steps)
            # replace the start only if they are better than the start
            w_lin, sigma_lin = linear_weights(T_mass, T_syst)
            with torch.no_grad():
                hard_lin = profile(T_mass, T_syst, linear(w_lin))["sigma"]
            logger.info(
                f"{prefix}linear-moment optimum: sigma_mt={sigma_lin:.4g} (mean only), hard sigma_mt={hard_lin:.4g}, "
                f"weights " + " ".join(f"{n}={x:+.4g}" for n, x in zip(names, w_lin.tolist())),
            )
            history = [{"start": "linear", "step": -1, "sigma_lin": sigma_lin, "hard_sigma_mt": hard_lin,
                        "weights": dict(zip(names, w_lin.cpu().tolist()))}]
            tail_begin = int(self.comb_steps * (1.0 - self.comb_tail_fraction))
            tail_hard = []
            v = w_lin.clone().to(device).requires_grad_(True)
            optimizer = torch.optim.Adam([v], lr=self.comb_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.comb_steps, 1), eta_min=self.comb_lr_min,
            )
            for step in range(self.comb_steps + 1):
                w = v / v.norm()
                s_mass = [T @ w for T in T_mass]
                s_syst = {
                    src: {d: (T @ w, ew) for d, (T, ew) in samples.items()}
                    for src, samples in T_syst.items()
                }
                sigma2 = _soft_sigma2(
                    [s_mass[i] for i in slope_points], slope_masses, s_mass[nominal_class], s_syst,
                    self.lik_events, self.lik_bins,
                )
                if step % self.comb_eval_every == 0 or step == self.comb_steps:
                    w_det = w.detach().clone()
                    with torch.no_grad():
                        hard = profile(T_mass, T_syst, linear(w_det))["sigma"]
                    if step >= tail_begin:
                        tail_hard.append(hard)
                    record = {
                        "start": "linear", "step": step, "lr": optimizer.param_groups[0]["lr"],
                        "soft_sigma_mt": float(sigma2.detach().clamp(min=0).sqrt()), "hard_sigma_mt": hard,
                        "weights": dict(zip(names, w_det.cpu().tolist())),
                    }
                    history.append(record)
                    logger.info(
                        f"{prefix}combination step {step}: soft sigma_mt={record['soft_sigma_mt']:.4g}, "
                        f"hard sigma_mt={hard:.4g}, weights " +
                        " ".join(f"{n}={x:+.4g}" for n, x in record["weights"].items()),
                    )
                if step == self.comb_steps:
                    break
                optimizer.zero_grad()
                sigma2.backward()
                optimizer.step()
                scheduler.step()

            tail_mean = float(np.mean(tail_hard))
            best_w, best_hard, choice = (w_det, tail_mean, "refined") if tail_mean < hard_lin else \
                (w_lin, hard_lin, "linear-moment optimum")
            logger.info(
                f"{prefix}combination: refined mean hard sigma_mt over the last {len(tail_hard)} evaluations "
                f"{tail_mean:.4g} +- {np.std(tail_hard):.2g} (spread), linear-moment optimum {hard_lin:.4g}, "
                f"selected {choice}, weights " + " ".join(f"{n}={x:+.4g}" for n, x in zip(names, best_w.tolist())),
            )
            return best_w, history

        # k-fold fit and evaluation on the held-out folds, compared to mtfit on the same events
        comb_history, fold_weights, fold_liks, fold_baselines = [], [], [], []
        for k in range(n_folds):
            prefix = f"fold {k}: "
            w_k, history = fit_weights(*take(T_mass_pool, T_syst_pool, k, held_out=False), prefix)
            comb_history += [{"fold": k, **r} for r in history]
            with torch.no_grad():
                lik = profile(*take(T_mass_pool, T_syst_pool, k, held_out=True), linear(w_k))
                base = profile(*take(mass_pool, syst_pool, k, held_out=True), lambda X: X[:, 0].cpu().numpy())
            logger.info(
                f"{prefix}held-out combination: {fmt_lik(lik)}, held-out mtfit sigma_mt={base['sigma']:.4g}, "
                f"ratio {lik['sigma'] / base['sigma']:.3g}",
            )
            fold_weights.append(w_k)
            fold_liks.append(lik)
            fold_baselines.append(base)
        cv_sigma = np.array([lik["sigma"] for lik in fold_liks])
        cv_base = np.array([lik["sigma"] for lik in fold_baselines])
        cv_ratio = cv_sigma / cv_base
        logger.info(
            f"k-fold estimate of the combination ({n_folds} folds): sigma_mt={cv_sigma.mean():.4g} "
            f"+- {cv_sigma.std() / np.sqrt(n_folds):.2g}, mtfit {cv_base.mean():.4g} "
            f"+- {cv_base.std() / np.sqrt(n_folds):.2g}, ratio {cv_ratio.mean():.3g} "
            f"+- {cv_ratio.std() / np.sqrt(n_folds):.2g}",
        )

        # observables of the 2D fit: k-fold fit and evaluation on the held-out folds, the 2D fit of the mass and
        # syst observables, the mass observable alone, and the 2D fit of mtfit and the syst observable
        def linear_2d(W):
            return lambda T: (T @ W).cpu().numpy()

        fold_weights_2d, fold_liks_2d, cv_2d = [], [], {"2d": [], "mass": [], "mtfit_syst": []}
        for k in range(n_folds):
            prefix = f"fold {k}: "
            W_k, sigma_lin_2d, corr = weights_2d(*take(T_mass_pool, T_syst_pool, k, held_out=False))
            held_T = take(T_mass_pool, T_syst_pool, k, held_out=True)
            held_X = take(mass_pool, syst_pool, k, held_out=True)
            with torch.no_grad():
                lik_2d = profile_2d(*held_T, linear_2d(W_k))
                lik_mass = profile(*held_T, linear(W_k[:, 0]))
                lik_mtfit_syst = profile_2d(
                    *held_X, lambda X: np.stack([X[:, 0].cpu().numpy(), (comb.logits(X) @ W_k[:, 1]).cpu().numpy()], 1),
                )
            base = fold_baselines[k]["sigma"]
            logger.info(
                f"{prefix}2D observables: mean-only sigma_mt={sigma_lin_2d:.4g}, correlation {corr:+.3g}, weights "
                f"mass " + " ".join(f"{n}={x:+.4g}" for n, x in zip(names, W_k[:, 0].tolist())) +
                ", syst " + " ".join(f"{n}={x:+.4g}" for n, x in zip(names, W_k[:, 1].tolist())),
            )
            logger.info(
                f"{prefix}held-out 2D fit (mass x syst): {fmt_lik(lik_2d)}, ratio to mtfit "
                f"{lik_2d['sigma'] / base:.3g}, to the 1D combination {lik_2d['sigma'] / fold_liks[k]['sigma']:.3g}; "
                f"mass observable alone sigma_mt={lik_mass['sigma']:.4g}, 2D fit mtfit x syst "
                f"sigma_mt={lik_mtfit_syst['sigma']:.4g}",
            )
            fold_weights_2d.append(W_k)
            fold_liks_2d.append(lik_2d)
            for key, lik in (("2d", lik_2d), ("mass", lik_mass), ("mtfit_syst", lik_mtfit_syst)):
                cv_2d[key].append(lik["sigma"])
        cv_2d = {key: np.array(v) for key, v in cv_2d.items()}
        logger.info(
            f"k-fold estimate of the 2D fit ({n_folds} folds): sigma_mt={cv_2d['2d'].mean():.4g} "
            f"+- {cv_2d['2d'].std() / np.sqrt(n_folds):.2g}, ratio to mtfit {(cv_2d['2d'] / cv_base).mean():.3g} "
            f"+- {(cv_2d['2d'] / cv_base).std() / np.sqrt(n_folds):.2g}, to the 1D combination "
            f"{(cv_2d['2d'] / cv_sigma).mean():.3g} +- {(cv_2d['2d'] / cv_sigma).std() / np.sqrt(n_folds):.2g}; "
            f"mass observable alone {cv_2d['mass'].mean():.4g}, 2D fit mtfit x syst {cv_2d['mtfit_syst'].mean():.4g}",
        )

        # scan of the number of bins: held-out estimates with the k-fold weights, mean and its uncertainty over the
        # folds, the noise is the quadratic mean over the sources of the expected noise of the prefit impacts
        # (it grows with the number of bins, combine does not subtract it)
        def fold_mean(values):
            values = np.array(values)
            return f"{values.mean():.4g} +- {values.std() / np.sqrt(len(values)):.2g}"

        def scan_line(label, liks):
            noise = [np.sqrt(np.mean(np.square(list(lik["noise"].values())))) for lik in liks]
            return (
                f"{label}: sigma_mt={fold_mean([lik['sigma'] for lik in liks])}, "
                f"raw {fold_mean([lik['sigma_raw'] for lik in liks])}, "
                f"stat {fold_mean([lik['sigma_stat'] for lik in liks])}, noise {fold_mean(noise)}"
            )

        bin_scan = {}
        for n_bins in self.scan_bins:
            liks = {"mtfit": [], "combination": []}
            for k in range(n_folds):
                with torch.no_grad():
                    liks["mtfit"].append(profile(
                        *take(mass_pool, syst_pool, k, held_out=True), lambda X: X[:, 0].cpu().numpy(),
                        n_bins=n_bins,
                    ))
                    liks["combination"].append(profile(
                        *take(T_mass_pool, T_syst_pool, k, held_out=True), linear(fold_weights[k]), n_bins=n_bins,
                    ))
            ratio = [c["sigma"] / b["sigma"] for c, b in zip(liks["combination"], liks["mtfit"])]
            logger.info(
                f"bin scan, {n_bins} bins: " + "; ".join(scan_line(key, v) for key, v in liks.items()) +
                f"; ratio {fold_mean(ratio)}",
            )
            bin_scan[str(n_bins)] = {key: [lik["sigma"] for lik in v] for key, v in liks.items()}
        for bins in self.scan_bins_2d:
            liks = {"mass x syst": [], "mtfit x syst": []}
            for k in range(n_folds):
                W_k = fold_weights_2d[k]
                with torch.no_grad():
                    liks["mass x syst"].append(profile_2d(
                        *take(T_mass_pool, T_syst_pool, k, held_out=True), linear_2d(W_k), bins=bins,
                    ))
                    liks["mtfit x syst"].append(profile_2d(
                        *take(mass_pool, syst_pool, k, held_out=True),
                        lambda X: np.stack([X[:, 0].cpu().numpy(), (comb.logits(X) @ W_k[:, 1]).cpu().numpy()], 1),
                        bins=bins,
                    ))
            ratio = [lik["sigma"] / base for lik, base in zip(liks["mass x syst"], cv_base)]
            logger.info(
                f"bin scan, {bins[0]}x{bins[1]} bins: " + "; ".join(scan_line(key, v) for key, v in liks.items()) +
                f"; ratio of mass x syst to mtfit ({self.lik_bins} bins) {fold_mean(ratio)}",
            )
            bin_scan[f"{bins[0]}x{bins[1]}"] = {key: [lik["sigma"] for lik in v] for key, v in liks.items()}

        # final weights: mean of the fold weights with aligned signs (the sign of the observable is irrelevant)
        w_final = torch.stack([w * torch.sign(torch.dot(w, fold_weights[0])) for w in fold_weights]).mean(dim=0)
        comb.weights = (w_final / w_final.norm()).cpu()
        with torch.no_grad():
            levels = torch.linspace(0.0, 1.0, 1001, dtype=T_nom.dtype, device=device)
            comb.quantiles = torch.quantile(T_mass_pool[nominal_class] @ comb.weights.to(device), levels).cpu()

        # evaluation on the pooled comb and test parts: mtfit and the score of each head alone (not fitted on these
        # events), and the combination with the final weights (fitted on these events, biased low, the k-fold
        # estimate above is the unbiased one)
        baseline_lik = profile(mass_pool, syst_pool, lambda X: X[:, 0].cpu().numpy(), self.n_boot)
        logger.info(f"baseline mtfit on the comb and test parts: {fmt_lik(baseline_lik)}")
        score_liks = {}
        with torch.no_grad():
            for i, name in enumerate(names):
                score_liks[name] = profile(
                    mass_pool, syst_pool, lambda X: comb.logits(X)[:, i].cpu().numpy(), self.n_boot,
                )
                logger.info(
                    f"score of head {name} on the comb and test parts: {fmt_lik(score_liks[name])}, "
                    f"ratio to mtfit {score_liks[name]['sigma'] / baseline_lik['sigma']:.3g}",
                )
            comb_lik = profile(mass_pool, syst_pool, lambda X: comb(X).reshape(-1).cpu().numpy(), self.n_boot)
            u_nom = comb(mass_pool[nominal_class]).reshape(-1).cpu().numpy()
        logger.info(
            f"combination with the final weights on the comb and test parts (used in the fit): {fmt_lik(comb_lik)}, "
            f"ratio to mtfit {comb_lik['sigma'] / baseline_lik['sigma']:.3g}, weights " +
            " ".join(f"{n}={v:+.4g}" for n, v in zip(names, comb.weights.tolist())),
        )

        # final weights of the 2D observables: mean of the fold weights, the sign of the mass observable is fixed by
        # w . mu_mt > 0, the syst observable is aligned with the first fold
        W_final = torch.stack([
            torch.stack([W[:, 0], W[:, 1] * torch.sign(torch.dot(W[:, 1], fold_weights_2d[0][:, 1]))], dim=1)
            for W in fold_weights_2d
        ]).mean(dim=0)
        comb.weights_2d = (W_final / W_final.norm(dim=0, keepdim=True)).cpu()
        with torch.no_grad():
            t_nom_2d = T_mass_pool[nominal_class] @ comb.weights_2d.to(device)
            comb.quantiles_2d = torch.stack([torch.quantile(t_nom_2d[:, i], levels) for i in range(2)]).cpu()
            lik_2d_final = profile_2d(
                mass_pool, syst_pool, lambda X: comb.observables_2d(X).cpu().numpy(), self.n_boot,
            )
            u_nom_2d = comb.observables_2d(mass_pool[nominal_class]).cpu().numpy()
        logger.info(
            f"2D fit (mass x syst, {self.bins_2d[0]} x {self.bins_2d[1]} bins) with the final weights on the comb and "
            f"test parts (used in the fit): {fmt_lik(lik_2d_final)}, ratio to mtfit "
            f"{lik_2d_final['sigma'] / baseline_lik['sigma']:.3g}, to the 1D combination "
            f"{lik_2d_final['sigma'] / comb_lik['sigma']:.3g}, correlation of the observables "
            f"{np.corrcoef(u_nom_2d.T)[0, 1]:+.3g}, weights mass " +
            " ".join(f"{n}={v:+.4g}" for n, v in zip(names, comb.weights_2d[:, 0].tolist())) +
            ", syst " + " ".join(f"{n}={v:+.4g}" for n, v in zip(names, comb.weights_2d[:, 1].tolist())),
        )

        # ten percentile bins of the output for the nominal events (close to equal bins by construction)
        edges = np.quantile(u_nom, np.linspace(0, 1, 11))
        edges[0], edges[-1] = 0.0, 1.0
        edges = [float(e) for e in edges]
        logger.info(f"percentile binning (10 bins) for mtop {self.mass_points[nominal_class]}: {edges}")

        output.touch()
        torch.save(
            {
                "names": names,
                "net": best_state,
                "hidden": self.hidden,
                "mean": mean,
                "std": std,
                "t_mean": comb.t_mean,
                "t_std": comb.t_std,
                "weights": comb.weights,
                "quantiles": comb.quantiles,
                "weights_2d": comb.weights_2d,
                "quantiles_2d": comb.quantiles_2d,
                "mass_points": self.mass_points,
            },
            get_path(output.child("model.pt", type="f")),
        )
        output.child("history.json", type="f").dump(
            {
                "mass_points": self.mass_points, "names": names, "weights": comb.weights.tolist(),
                "baseline_mtfit_lik": baseline_lik, "score_liks": score_liks, "combination_lik": comb_lik,
                "binning": edges, "classifier": clf_history, "combination": comb_history,
                "kfold": {
                    "sigma": cv_sigma.tolist(), "mtfit": cv_base.tolist(), "liks": fold_liks,
                    "weights": [w.tolist() for w in fold_weights],
                },
                "bin_scan": bin_scan,
                "obs_2d": {
                    "bins": self.bins_2d, "weights": comb.weights_2d.tolist(), "lik": lik_2d_final,
                    "kfold": {
                        **{key: v.tolist() for key, v in cv_2d.items()}, "liks": fold_liks_2d,
                        "weights": [W.tolist() for W in fold_weights_2d],
                    },
                },
            },
            formatter="json",
        )

    def open_model(self, target: law.FileSystemDirectoryTarget) -> dict[str, Any]:
        checkpoint = torch.load(get_path(target.child("model.pt", type="f")), map_location="cpu", weights_only=False)
        net = _build_net(checkpoint["hidden"], len(checkpoint["names"]))
        net.load_state_dict(checkpoint["net"])
        comb = _Combination(
            net, checkpoint["t_mean"], checkpoint["t_std"], checkpoint["weights"], checkpoint["quantiles"],
            checkpoint.get("weights_2d"), checkpoint.get("quantiles_2d"),
        )
        comb.eval()
        return {"body": comb, "mean": checkpoint["mean"], "std": checkpoint["std"]}

    def evaluate(
            self,
            task: law.Task,
            events: ak.Array,
            models: list[Any],
            fold_indices: ak.Array,
            events_used_in_training: bool = False,
    ) -> ak.Array:
        X, valid = self._features(events)
        fold_indices = ak.to_numpy(fold_indices)
        output = np.full(len(events), -1.0, dtype=np.float32)
        obs_2d = np.full((len(events), 2), -1.0, dtype=np.float32)
        for fold, model in enumerate(models):
            mask = valid & (fold_indices == fold)
            if not np.any(mask):
                continue
            with torch.no_grad():
                X_fold = torch.as_tensor((X[mask] - model["mean"]) / model["std"])
                output[mask] = model["body"](X_fold).numpy().reshape(-1)
                if model["body"].weights_2d is not None:
                    obs_2d[mask] = model["body"].observables_2d(X_fold).numpy()

        # the cdf outputs are kept below 1 so that they stay in the first ten bins, events with a large fitted top
        # mass (first feature is log(mtfit)) get the output of the 11th bin
        tail = valid & (np.exp(X[:, 0]) > self.mtfit_max)
        output[valid] = np.minimum(output[valid], np.float32(1.0 - 1e-6))
        output[tail] = self.tail_output

        # 2D observables and their unrolled percentile bins (mass bins inner), the tail events in the last bin
        n_a, n_b = self.bins_2d
        has_2d = valid & (obs_2d[:, 0] >= 0)
        obs_2d[has_2d] = np.minimum(obs_2d[has_2d], np.float32(1.0 - 1e-6))
        unrolled = np.full(len(events), -1.0, dtype=np.float32)
        i_2d = np.floor(obs_2d[has_2d] * np.array([n_a, n_b], dtype=np.float32)).astype(np.int64)
        unrolled[has_2d] = i_2d[:, 0] + n_a * i_2d[:, 1]
        obs_2d[tail & has_2d] = self.tail_output
        unrolled[tail & has_2d] = n_a * n_b

        events = set_ak_column(events, f"{self.cls_name}.output", output)
        # column slices are strided views, parquet needs contiguous buffers
        events = set_ak_column(events, f"{self.cls_name}.mass", np.ascontiguousarray(obs_2d[:, 0]))
        events = set_ak_column(events, f"{self.cls_name}.syst", np.ascontiguousarray(obs_2d[:, 1]))
        events = set_ak_column(events, f"{self.cls_name}.unrolled", unrolled)
        return events


# usable derivations
mass_classifier_dnn = MassClassifierDNN.derive("mass_classifier_dnn", cls_dict={"folds": 3})
mass_classifier_dnn_1fold = MassClassifierDNN.derive("mass_classifier_dnn_1fold", cls_dict={"folds": 1})
# short test run with one fold, settings via cls_dict avoid --ml-model-settings
mass_classifier_dnn_test = MassClassifierDNN.derive("mass_classifier_dnn_test", cls_dict={
    "folds": 1, "epochs": 100,
})
