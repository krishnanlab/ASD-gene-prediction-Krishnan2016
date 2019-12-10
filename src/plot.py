#!/usr/bin/env python3
import numpy as np
import os
import glob
import argparse
import matplotlib
from utils import load_weighted_labels
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval(file):
    data = np.loadtxt(file, skiprows=1, delimiter="\t")
    auroc = data[0]
    # auprc = data[1]
    return auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--results", required=True, help="results directory")
    parser.add_argument(
        "-e", "--evaluations", required=True, nargs="+",
        help="evaluations to plot")
    args = parser.parse_args()

    plots_dir = os.path.join(args.results, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    results = {evaluation: [] for evaluation in args.evaluations}

    for evaluation in args.evaluations:
        eval_dir = os.path.join(args.results, "evaluations", evaluation)
        files = os.listdir(eval_dir)
        for file in files:
            results[evaluation].append(load_eval(os.path.join(eval_dir, file)))

    plt.boxplot(results.values(), labels=results.keys(), vert=False)
    plt.xlabel("auROC")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "auROC.pdf"))


