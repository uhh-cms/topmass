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
    "tt_*_powheg/nominal/calib__default/sel__default/"
    "red__cf_default/prod__kinFitMatch/{version}/columns_*.parquet"
)

BASE_PATH_CAT = (
    "/data/dust/user/griesinl/aj_store/analysis_aj/cf.ProduceColumns/2017_v9/"
    "tt_*_powheg/nominal/calib__default/sel__default/"
    "red__cf_default/prod__default/{version}/columns_*.parquet"
)


def load_data(version: str):
    logger.info(f"Loading dataset version: {version}")

    path = BASE_PATH.format(version=version)
    path_cat = BASE_PATH_CAT.format(version=version)

    mtfit = ak.from_parquet(path, columns=["events", "FitTop1"])
    avg_W = ak.from_parquet(path, columns=["events", "RecoW1", "RecoW2"])
    jets = ak.from_parquet(path, columns=["events", "FitJet.reco.pt"])

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

    return mtfit, avg_W, jets, categories, weights


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


def compute_variables(mtfit, avg_W, jets, mask):
    top1 = build_4vec(mtfit, mtfit.FitTop1)
    mtfit_flat = top1[mask].mass

    recoW1 = build_4vec(avg_W, avg_W.RecoW1)
    recoW2 = build_4vec(avg_W, avg_W.RecoW2)
    avg_W_flat = 0.5 * (recoW1[mask].mass + recoW2[mask].mass)

    j = jets.FitJet.reco[mask]
    rbq = (j[:, 0].pt + j[:, 1].pt) / ak.sum(j[:, 2:6].pt, axis=1)

    return {"mtfit": mtfit_flat, "avg_W_mass": avg_W_flat, "reco_R_bq": rbq}


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


def get_2d_edges(x, y, w, nx=6, ny=3):
    x = ak.to_numpy(x)
    y = ak.to_numpy(y)
    w = ak.to_numpy(w)

    x_edges = quantile_edges(x, w, nx)
    y_edges = quantile_edges(y, w, ny)

    return x_edges, y_edges


def compute_edges(arr, w=None, nbins=8):
    percentiles = np.linspace(0, 100, nbins + 1)

    if w is None:
        return np.percentile(arr, percentiles)

    return weighted_percentiles(arr, w, percentiles)


def format_edges(edges):
    return "[" + ", ".join(f"{x:.6g}" for x in edges) + "]"


# main function
def main(args):
    mtfit, avg_W, jets, categories, weights = load_data(args.version)

    mask = categories.category_ids == 502
    goodmask = ak.any(mask, axis=1)

    logger.info(f"Selected events: {ak.sum(goodmask)} / {len(goodmask)}")

    w = build_weight(weights)[goodmask]
    vars_dict = compute_variables(mtfit, avg_W, jets, goodmask)

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
        vars_dict["avg_W_mass"],
        w,
        nx=args.nbins_x,
        ny=args.nbins_y,
    )

    logger.info(f"nx × ny  : {args.nbins_x} × {args.nbins_y}")
    logger.info(f"X edges mtfit  : {format_edges(x_edges)}")
    logger.info(f"Y edges avg_W_mass  : {format_edges(y_edges)}")
    logger.info("=========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, default="v1_TopMass")

    parser.add_argument("--vars", type=str, default="all")

    parser.add_argument("--nbins", type=int, default=8, help="1D binning only")

    parser.add_argument("--nbins-x", type=int, default=6, help="mtfit bins for 2D")

    parser.add_argument("--nbins-y", type=int, default=3, help="avg_W bins for 2D")

    args = parser.parse_args()
    main(args)
