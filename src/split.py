#!/usr/bin/env python3
import argparse
import os
from sklearn.model_selection import KFold
from utils import load_weighted_labels


def split(labels_files, n_trials, k, output_dir):
    for labels_file in labels_files:
        IDs, labels, weights = load_weighted_labels(labels_file)

        # remove IDs labeled 0
        IDs, labels, weights = zip(*[
            [ID, label, weight] for ID, label, weight
            in zip(IDs, labels, weights) if label != 0])

        for i in range(n_trials):
            train_indices, test_indices = zip(
                *KFold(n_splits=k, shuffle=True).split(IDs, labels))
            for j in range(k):
                output_file = "{}_trial{}_fold{}.tsv".format(
                    os.path.basename(labels_file).split(".")[0], i + 1, j + 1)
                output_file = os.path.join(output_dir, output_file)

                with open(output_file, "w") as f:
                    for index in train_indices[j]:
                        f.write("{}\t{}\t{}\n".format(
                            IDs[index], labels[index], weights[index],))

                    for index in test_indices[j]:
                        f.write("{}\t{}\t{}\n".format(
                            IDs[index], 0, weights[index]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels", required=True, nargs="+",
                        help="labels file(s)")
    parser.add_argument("-n", "--n_trials", type=int, default=1,
                        help="number of cross-validation trials to perform")
    parser.add_argument("-k", "--k_folds", type=int, default=3,
                        help="number of folds for k-fold cross-validation")
    parser.add_argument("-o", "--output", required=True,
                        help="output directory for train/test splits")
    args = parser.parse_args()

    for file in args.labels:
        if not os.path.exists(file):
            parser.error(file + " does not exist.")

    os.makedirs(args.output, exist_ok=True)

    split(args.labels, args.n_trials, args.k_folds, args.output)
