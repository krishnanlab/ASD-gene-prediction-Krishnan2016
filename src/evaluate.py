#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import load_predictions, load_labels, load_weighted_labels
import argparse
import os


def evaluate(predictions_file, labels_files, output_dir):

    IDs = []
    labels = []
    weights = []
    for labels_file in labels_files:
        file_IDs, file_labels, file_weights = \
            load_weighted_labels(labels_file)
        IDs += file_IDs
        labels += list(file_labels)
        weights += list(file_weights)
    labels = np.array(labels)
    weights = np.array(weights)

    predicted_IDs, probs = load_predictions(predictions_file)

    evaluate_IDs = []
    evaluate_labels = []
    evaluate_probs = []

    for i, ID in enumerate(predicted_IDs):
        # only evaluate if...
        # ID has weight = 1 in the original labels
        # and ID is not ambiguous in the original labels
        if (ID in IDs and weights[IDs.index(ID)] == 1
                and labels[IDs.index(ID)] != 0):
            evaluate_IDs.append(ID)
            evaluate_labels.append(labels[IDs.index(ID)])
            evaluate_probs.append(probs[i])

    evaluate_IDs = np.array(evaluate_IDs)
    evaluate_labels = np.array(evaluate_labels)
    evaluate_probs = np.array(evaluate_probs)

    try:
        sort_index = evaluate_probs.argsort()[::-1]
        evaluate_IDs = evaluate_IDs[sort_index]
        evaluate_labels = evaluate_labels[sort_index]
        evaluate_probs = evaluate_probs[sort_index]

        prior = sum(evaluate_labels == 1) / len(evaluate_labels)
        ranks = [10, 20, 50]

        auROC = roc_auc_score(evaluate_labels, evaluate_probs)
        auPRC = np.log2(
            average_precision_score(evaluate_labels, evaluate_probs)
            / prior)

        # precision = TP / (TP + FP)
        precisions = \
            [np.log2((sum(evaluate_labels[:rank] == 1) / rank) / prior)
                for rank in ranks]

        # recall = TP / (TP + FN)
        recalls = \
            [(sum(evaluate_labels[:rank] == 1) / sum(evaluate_labels == 1))
                for rank in ranks]

        precision_headers = \
            ["log(P@r{}/prior)".format(rank) for rank in ranks]
        recall_headers = ["R@r{}".format(rank) for rank in ranks]

        values = [auROC] + [auPRC] + precisions + recalls
        headers = \
            ["auROC"] + ["auPRC"] + precision_headers + recall_headers

        output_file = \
            os.path.join(output_dir, os.path.basename(predictions_file))
        with open(output_file, "w") as f:
            blank_string = "\t".join(["{}"] * (2 * len(ranks) + 2)) + "\n"
            f.write(blank_string.format(*headers))
            f.write(blank_string.format(*values))

    except ValueError as e:
        print("ValueError:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--predictions", required=True, help="predictions file")
    parser.add_argument(
        "-l", "--labels", required=True, nargs="+", help="labels file(s)")
    parser.add_argument(
        "-o", "--output", required=True,
        help="output directory for cross-validation metrics")

    args = parser.parse_args()

    should_exist = [args.predictions] + args.labels
    for filename in should_exist:
        if filename is not None:
            if not os.path.exists(filename):
                parser.error(filename + " does not exist")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    evaluate(args.predictions, args.labels, args.output)
