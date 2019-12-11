#!/usr/bin/env python3
import argparse
import os
from utils import load_weighted_labels, load_predictions


def unsplit(split_files, pred_file, output_dir):
    unsplit_pred = {}

    split_IDs = []
    split_labels = []
    for split_file in split_files:
        split_IDs_file, split_labels_file, _ = \
            load_weighted_labels(split_file)
        split_IDs += split_IDs_file
        split_labels += list(split_labels_file)
    split = dict(zip(split_IDs, split_labels))

    pred = dict(zip(*load_predictions(pred_file)))

    for ID in pred:
        if ID in split:
            if split[ID] == 0:
                if ID in unsplit_pred:
                    unsplit_pred[ID].append(pred[ID])
                else:
                    unsplit_pred[ID] = [pred[ID]]
        else:
            if ID in unsplit_pred:
                unsplit_pred[ID].append(pred[ID])
            else:
                unsplit_pred[ID] = [pred[ID]]

    for ID in unsplit_pred:
        unsplit_pred[ID] = sum(unsplit_pred[ID]) / len(unsplit_pred[ID])

    output_file = os.path.join(output_dir, os.path.basename(pred_file))
    with open(output_file, "w") as f:
        for ID, prob in unsplit_pred.items():
            f.write("{}\t{}\n".format(ID, prob))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--splits", required=True, nargs="+",
                        help="splits files")
    parser.add_argument("-p", "--predictions", required=True,
                        help="predictions file")
    parser.add_argument("-o", "--output", required=True,
                        help="output directory for combined predictions")
    args = parser.parse_args()

    should_exist = args.splits + [args.predictions]
    for file in should_exist:
        if not os.path.exists(file):
            parser.error(file + " does not exist.")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    unsplit(args.splits, args.predictions, args.output)
