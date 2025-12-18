import argparse
from pathlib import Path
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt

# Import your existing plotting function and od classes
from alljets.plotting.plot_two_variables import plot_unmatched_matched
import order as od  # Adjust accordingly

def load_histograms(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Compare two categories from one pickle histogram file.")

    parser.add_argument("--pickle", type=Path, required=True, help="Path to pickle file containing all categories.")
    parser.add_argument("--cat1-name", required=True, help="Name of first category (must be a top-level key in pickle).")
    parser.add_argument("--cat2-name", required=True, help="Name of second category (must be a top-level key in pickle).")

    parser.add_argument("--process", required=True, help="Process name to extract histograms from both categories.")
    parser.add_argument("--variable", required=True, help="Variable name to project and compare.")

    parser.add_argument("--output", type=Path, required=True, help="Output path for the combined plot image.")

    parser.add_argument("--density", action="store_true", help="Apply density normalization.")
    parser.add_argument("--shape-norm", action="store_true", help="Apply shape normalization.")
    parser.add_argument("--yscale", default="linear", choices=["linear", "log"], help="Y-axis scale.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")

    args = parser.parse_args()

    # Load the full pickle file (all categories)
    hist = load_histograms(args.pickle)

    import IPython
     # Validate category and process existence
    cat_values = list(hist.axes["category"])
    proc_values = list(hist.axes["process"])

    if args.cat1_name not in cat_values:
        raise ValueError(f"Category '{args.cat1_name}' not found in histogram (available: {cat_values})")
    if args.cat2_name not in cat_values:
        raise ValueError(f"Category '{args.cat2_name}' not found in histogram (available: {cat_values})")
    if args.process not in proc_values:
        raise ValueError(f"Process '{args.process}' not found in histogram (available: {proc_values})")


    from alljets.config.analysis_aj import analysis_aj  # or wherever analysis_aj is defined

    config_inst = analysis_aj.configs.get("2017_v9")  # or another registered config name
    # IPython.embed()


    category_inst = od.Category(name=f"{args.cat1_name}_vs_{args.cat2_name}")
    variable_inst = config_inst.variables.get(args.variable)
    hist_dict = {str(cat): hist[{"category": cat}] for cat in hist.axes["category"]}

    # Call plotting function
    fig, axes = plot_unmatched_matched(
        hists=hist_dict,
        config_inst=config_inst,
        category_inst=category_inst,
        variable_insts=[variable_inst],
        density=args.density,
        shape_norm=args.shape_norm,
        yscale=args.yscale,
    )

    # Save and optionally show
    fig.savefig(args.output)
    print(f"✅ Plot saved to: {args.output}")
    if args.show:
        fig.show()

if __name__ == "__main__":
    main()
