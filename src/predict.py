#!/usr/bin/env python3
import numpy as np
import liblinearutil as ll
from utils import (load_edgelist, load_adjacency, load_numpy,
                   load_feature_row_file, load_weighted_labels,
                   get_class_data)
import argparse
import os


def predict(input_file, input_type, labels_files, output_dir,
            fill_diagonal=None, rows_file=None, test=False, save_model=False):
    if input_type == "edgelist":
        data, IDs = load_edgelist(input_file, fill_diagonal)
    elif input_type == "adjacency":
        data, IDs = load_adjacency(input_file, rows_file, fill_diagonal)
    elif input_type == "numpy":
        data, IDs = load_numpy(input_file, rows_file, fill_diagonal)
    else:
        data, IDs = load_feature_row_file(input_file, rows_file)

    labeled_IDs = []
    labels = []
    weights = []
    for labels_file in labels_files:
        file_labeled_IDs, file_labels, file_weights = \
            load_weighted_labels(labels_file)
        labeled_IDs += file_labeled_IDs
        labels += list(file_labels)
        weights += list(file_weights)
    labels = np.array(labels)
    weights = np.array(weights)

    train_data, _, train_labels, train_weights = get_class_data(
        data, IDs, labeled_IDs, labels, weights, classes=[1, -1])

    print("Training...")
    model = ll.train(train_weights, train_labels, train_data, "-s 7")

    print("Predicting...")
    if test:
        test_data, test_IDs, test_labels, _ = get_class_data(
            data, IDs, labeled_IDs, labels, weights, classes=[0])
    else:
        test_data = data
        test_IDs = IDs

    pred_labels, _, probs = \
        ll.predict([], test_data, model, "-b 1")

    # keep only the probability that the gene is positive
    probs = np.array([sublist[0] for sublist in probs])

    print("\nSaving predictions...")
    output_file = os.path.join(
        output_dir,
        "_".join(os.path.basename(labels_files[0]).split("_")[1:]))
    with open(output_file, "w") as f:
        for ID, label, prob in zip(test_IDs, pred_labels, probs):
            f.write("{0}\t{1}\n" .format(ID, prob))

    if save_model:
        print("\nSaving model...")
        model_file = os.path.basename(labels_file).split(".")[0] + ".model"
        ll.save_model(os.path.join(output_dir, model_file), model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    input_file = parser.add_mutually_exclusive_group(required=True)
    input_file.add_argument(
        "-e", "--edgelist", help="edgelist file describing gene network")
    input_file.add_argument(
        "-a", "--adjacency",
        help="adjacency matrix file describing gene network")
    input_file.add_argument(
        "-n", "--numpy",
        help="numpy adjacency matrix file describing gene network")
    input_file.add_argument(
        "-f", "--features",
        help="libSVM-formatted feature file describing gene network")

    parser.add_argument(
        "-d", "--fill_diagonal", type=float,
        help="value to fill along the diagonal")
    parser.add_argument(
        "-r", "--rows", help="list of gene IDs for rows in feature file")
    parser.add_argument(
        "-l", "--labels", required=True, nargs="+", type=str,
        help="labels file(s) including weights")
    parser.add_argument(
        "-o", "--output", required=True,
        help="output directory for predictions")
    parser.add_argument(
        "-s", "--save", action="store_true",
        help="save model file to output directory")
    parser.add_argument(
        "-t", "--test", action="store_true",
        help="only predict on test labels")

    args = parser.parse_args()

    if args.features is not None:
        if args.rows is None:
            parser.error("Must supply row gene IDs if using feature file")
        else:
            input_file = args.features
            input_type = "features"
    elif args.adjacency is not None:
        if args.rows is None:
            parser.error(
                "Must supply row gene IDs if using adjacency matrix file")
        else:
            input_file = args.adjacency
            input_type = "adjacency"
    elif args.numpy is not None:
        if args.rows is None:
            parser.error(
                "Must supply row gene IDs if using numpy matrix file")
        else:
            input_file = args.numpy
            input_type = "numpy"
    else:  # input must be edgelist
        input_file = args.edgelist
        input_type = "edgelist"

    should_exist = args.labels + [
        args.edgelist, args.adjacency, args.numpy, args.features, args.rows]
    for filename in should_exist:
        if filename is not None:
            if not os.path.exists(filename):
                parser.error(filename + " does not exist")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    predict(
        input_file, input_type, args.labels, args.output,
        args.fill_diagonal, args.rows, args.test, args.save)
