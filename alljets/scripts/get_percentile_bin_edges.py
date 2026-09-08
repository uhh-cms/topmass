import argparse
import logging
import numpy as np
import awkward as ak
import vector

vector.register_awkward()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("topmass")


# Loading data
BASE_PATH = (
    "/data/dust/user/griesinl/aj_store/analysis_aj/cf.ProduceColumns/2017_v9/"
    "tt_*_powheg/nominal/calib__default/sel__default__steps_json_met_filter_pv_67b7c17ff4/"
    "red__cf_default/prod__kinFitMatch/{version}/columns_*.parquet"
)

BASE_PATH_CAT = (
    "/data/dust/user/griesinl/aj_store/analysis_aj/cf.ProduceColumns/2017_v9/"
    "tt_*_powheg/nominal/calib__default/sel__default__steps_json_met_filter_pv_67b7c17ff4/"
    "red__cf_default/prod__default/{version}/columns_*.parquet"
)


def load_data(version: str):
    logger.info(f"Loading dataset version: {version}")

    path = BASE_PATH.format(version=version)
    path_cat = BASE_PATH_CAT.format(version=version)

    precomputed = ak.from_parquet(
        path,
        columns=["events", "FitTopMass", "RecoWAvgMass", "RecoRbq"],
    )
    jets = ak.from_parquet(path, columns=["events", "FitJet.reco.{pt,eta,phi,mass}"])

    categories = ak.from_parquet(path_cat, columns=["events", "category_ids"])

    weights = ak.from_parquet(
        path_cat,
        columns=[
            "events",
            "btag_weight",
            "normalization_weight",
            "normalized_trig_weight",
            "normalized_pu_weight",
        ],
    )

    logger.info("Data loaded")

    return precomputed, jets, categories, weights


# helper functions
def build_4vec(df, field):
    return ak.zip(
        {
            "x": field.x,
            "y": field.y,
            "z": field.z,
            "t": field.t,
        },
        with_name="Momentum4D",
    )


def build_jet4vec(reco):
    return ak.zip(
        {
            "pt": reco.pt,
            "eta": reco.eta,
            "phi": reco.phi,
            "mass": reco.mass,
        },
        with_name="Momentum4D",
    )


def build_avg_R_bq(jets):
    """Calculate average R_bq using per-top matching"""
    reco = jets.FitJet.reco

    # Top 1: b-jet from first two, light quarks from indices 2,3
    R_bq_top1 = reco[:, 0].pt / (reco[:, 2].pt + reco[:, 3].pt)
    # Top 2: b-jet from indices 1, light quarks from indices 4,5
    R_bq_top2 = reco[:, 1].pt / (reco[:, 4].pt + reco[:, 5].pt)

    return (R_bq_top1 + R_bq_top2) / 2


def compute_variables(precomputed, jets, mask):
    mtfit_flat = precomputed.FitTopMass[mask]
    avg_W_flat = precomputed.RecoWAvgMass[mask]
    rbq = precomputed.RecoRbq[mask]

    # Add the new average R_bq calculation
    avg_rbq = build_avg_R_bq(jets)[mask]

    j = jets.FitJet.reco[mask]
    j4 = build_jet4vec(j)
    b_system = j4[:, 0] + j4[:, 1]
    q_system = j4[:, 2] + j4[:, 3] + j4[:, 4] + j4[:, 5]
    pt_b_vec, pt_q_vec = b_system.pt, q_system.pt

    rbq_vec = pt_b_vec / pt_q_vec
    rbq_vec_diff = pt_b_vec - pt_q_vec
    rbq_vec_rel_diff_sum = (pt_b_vec - pt_q_vec) / (pt_b_vec + pt_q_vec)
    rbq_vec_rel_diff_q = (pt_b_vec - pt_q_vec) / pt_q_vec

    return {
        "mtfit": mtfit_flat,
        "avg_W_mass": avg_W_flat,
        "reco_R_bq": rbq,
        "reco_avg_R_bq": avg_rbq,  # New variable
        "reco_R_bq_vec": rbq_vec,
        "reco_R_bq_vec_diff": rbq_vec_diff,
        "reco_R_bq_vec_rel_diff_sum": rbq_vec_rel_diff_sum,
        "reco_R_bq_vec_rel_diff_q": rbq_vec_rel_diff_q,
    }


def build_weight(weights):
    return (
        weights.btag_weight *
        weights.normalization_weight *
        weights.normalized_trig_weight *
        weights.normalized_pu_weight
    )


def weighted_percentiles(data, weights, percentiles):
    data = ak.to_numpy(data)
    weights = ak.to_numpy(weights)

    sorter = np.argsort(data)
    data = data[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(percentiles / 100.0, cdf, data)


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values)
    weights = np.asarray(weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, values)


def quantile_edges(x, w, n):
    qs = np.linspace(0, 1, n + 1)
    return weighted_quantile(x, w, qs)


def get_2d_edges(x, y, w, nx=6, ny=3, range_x=[0, 2000], range_y=[0, 2000]):
    x = ak.to_numpy(x)
    y = ak.to_numpy(y)
    w = ak.to_numpy(w)

    selected = (x >= range_x[0]) & (x <= range_x[1]) & (y >= range_y[0]) & (y <= range_y[1])
    print(np.sum(selected), len(x))

    x = x[selected]
    y = y[selected]
    w = w[selected]

    x_edges = quantile_edges(x, w, nx)
    y_edges = quantile_edges(y, w, ny)

    return x_edges, y_edges


def get_3d_edges(x, y, z, w, nx=4, ny=3, nz=2):
    x = ak.to_numpy(x)
    y = ak.to_numpy(y)
    z = ak.to_numpy(z)
    w = ak.to_numpy(w)

    x_edges = quantile_edges(x, w, nx)
    y_edges = quantile_edges(y, w, ny)
    z_edges = quantile_edges(z, w, nz)

    return x_edges, y_edges, z_edges


def compute_edges(arr, w=None, nbins=8):
    percentiles = np.linspace(0, 100, nbins + 1)

    if w is None:
        return np.percentile(arr, percentiles)

    return weighted_percentiles(arr, w, percentiles)


def format_edges(edges):
    return "[" + ", ".join(f"{x:.6g}" for x in edges) + "]"


# main function
def main(args):
    precomputed, jets, categories, weights = load_data(args.version)

    mask = categories.category_ids == 502
    goodmask = ak.any(mask, axis=1)

    logger.info(f"Selected events: {ak.sum(goodmask)} / {len(goodmask)}")

    w = build_weight(weights)[goodmask]
    vars_dict = compute_variables(precomputed, jets, goodmask)

    selected = args.vars
    if selected == "all":
        selected = list(vars_dict.keys())
    else:
        selected = selected.split(",")

    logger.info("========================================")
    logger.info("Binning summary")
    logger.info(f"Version : {args.version}")
    logger.info(f"Selected variables: {', '.join(selected)}")
    logger.info(f"Number of bins: {args.nbins}")
    logger.info("========================================")
    logger.info("[1D BINNING]")

    for v in selected:
        if v not in vars_dict:
            logger.warning(f"Unknown variable: {v}")
            continue

        edges = compute_edges(vars_dict[v], w, nbins=args.nbins)

        logger.info(f"Variable : {v}")
        logger.info(f"Percentile Edges    : {format_edges(edges)}")
        logger.info("=========================================")

    logger.info("[2D BINNING] mtfit ⊗ avg_W_mass")

    x_edges, y_edges = get_2d_edges(
        vars_dict["mtfit"],
        vars_dict["reco_R_bq"],
        w,
        nx=args.nbins_x,
        ny=args.nbins_y,
        range_x=[float(x) for x in args.range_x.split(",")],
        range_y=[float(y) for y in args.range_y.split(",")],
    )

    logger.info(f"nx × ny  : {args.nbins_x} × {args.nbins_y}")
    logger.info(f"X edges mtfit  : {format_edges(x_edges)}")
    logger.info(f"Y edges reco_R_bq : {format_edges(y_edges)}")
    logger.info("=========================================")

    rbq_vars = [
        "reco_R_bq",
        "reco_avg_R_bq",
        "reco_R_bq_vec",
        "reco_R_bq_vec_diff",
        "reco_R_bq_vec_rel_diff_sum",
        "reco_R_bq_vec_rel_diff_q",
    ]

    for rbq_var in rbq_vars:
        logger.info(f"[3D BINNING] mtfit ⊗ avg_W_mass ⊗ {rbq_var}")

        x3_edges, y3_edges, z3_edges = get_3d_edges(
            vars_dict["mtfit"],
            vars_dict["avg_W_mass"],
            vars_dict[rbq_var],
            w,
            nx=args.nbins_x3,
            ny=args.nbins_y3,
            nz=args.nbins_z3,
        )

        logger.info(f"nx × ny × nz  : {args.nbins_x3} × {args.nbins_y3} × {args.nbins_z3}")
        logger.info(f"X edges mtfit  : {format_edges(x3_edges)}")
        logger.info(f"Y edges avg_W_mass  : {format_edges(y3_edges)}")
        logger.info(f"Z edges {rbq_var}  : {format_edges(z3_edges)}")
        logger.info("=========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, default="v1_TopMass")

    parser.add_argument("--vars", type=str, default="all")

    parser.add_argument("--nbins", type=int, default=8, help="1D binning only")

    parser.add_argument("--nbins-x", type=int, default=6, help="mtfit bins for 2D")

    parser.add_argument("--nbins-y", type=int, default=3, help="avg_W bins for 2D")

    parser.add_argument("--range-x", type=str, default="0,2000", help="mtfit range for 2D")

    parser.add_argument("--range-y", type=str, default="0,2000", help="avg_W range for 2D")

    parser.add_argument("--nbins-x3", type=int, default=4, help="mtfit bins for 3D")

    parser.add_argument("--nbins-y3", type=int, default=3, help="avg_W bins for 3D")

    parser.add_argument("--nbins-z3", type=int, default=2, help="R_bq bins for 3D (all variants)")

    args = parser.parse_args()
    main(args)
