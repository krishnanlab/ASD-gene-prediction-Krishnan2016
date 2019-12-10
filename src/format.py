#!/usr/bin/env python3
import numpy as np
import argparse
import os
from utils import load_edgelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--network", required=True, help="path to network")
    parser.add_argument("-o", "--output", help="output directory")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        parser.error("{} does not exist".format(args.network))

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
    else:
        args.output = os.path.dirname(args.network)

    data, genes = load_edgelist(args.network)
    n_genes = len(genes)
    network = np.zeros((n_genes, n_genes))
    for i, item in enumerate(data):
        col_inds = [index - 1 for index in data[i].keys()]
        col_weights = list(data[i].values())
        network[i, col_inds] = col_weights

    basename = os.path.basename(args.network).split(".")[0]
    output_network = os.path.join(args.output, "{}.npy".format(basename))
    output_genes = os.path.join(args.output, "{}_genes.tsv".format(basename))

    np.save(output_network, network)
    np.savetxt(output_genes, genes, delimiter="\t", fmt="%s")


