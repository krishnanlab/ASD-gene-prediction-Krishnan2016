import numpy as np
import liblinearutil as ll
from sklearn.model_selection import KFold


# TODO: implement support for non-symmetric matrix
# should be loaded using either edgelist or adjacency
# (LL format should already work)
# will need to disable setting diagonal to 1 in each loading case
# should this even be the default?
# should we instead do this during preprocessing step?


def load_edgelist(edgelist_file, fill_diagonal=None):
    """
    Reads an edgelist and returns an adjacency dictionary and corresponding
    IDs.

    Parameters
    ----------
        edgelist_file : str
            Path to a tab-separated file with lines formatted like
            gene_1 \\t gene_2 \\t edge_weight.

    Returns
    -------
        data : list(dict)
            A list of feature vector dicts where elements are accessed by node
            ID.
        IDs : list(str)
            Gene IDs corresponding to each row of `data`.
    """
    data = []
    ID_map = {}
    current_node = 0

    with open(edgelist_file, "r") as f:
        for line in f:
            line = line.split("\t")
            if len(line) > 3:
                g1, g2, _, weight = line
            else:
                g1, g2, weight = line
            weight = float(weight)

            if g1 not in ID_map:
                ID_map[g1] = current_node
                # dict is indexed by ID_map + 1 because
                # LibLINEAR segfaults when first feature is 0
                data.append({})
                current_node += 1
            if g2 not in ID_map:
                ID_map[g2] = current_node
                data.append({})
                current_node += 1

            # check for duplicate edges and print warning
            if ID_map[g2] + 1 in data[ID_map[g1]]:
                print("Duplicate edge between {} and {}".format(g1, g2))

            data[ID_map[g1]][ID_map[g2] + 1] = weight
            data[ID_map[g2]][ID_map[g1] + 1] = weight

    IDs = sorted(ID_map, key=ID_map.get)

    # force self edges to be equal to diagonal value
    if fill_diagonal is not None:
        for node in range(current_node):
            data[node][node + 1] = fill_diagonal

    return data, IDs


def load_adjacency(adjacency_file, rows_file, fill_diagonal=None):
    """
    Reads and returns an adjacency matrix and corresponding IDs.

    Parameters
    ----------
        adjacency_file : str
            Path to a tab-separated adjacency matrix file.

    Returns
    -------
        data : list(dict)
            An adjacency matrix.
        IDs : list(str)
            Gene IDs corresponding to each row of `data`.
    """
    data = np.loadtxt(adjacency_file, delimiter="\t")

    if fill_diagonal is not None:
        np.fill_diagonal(data, fill_diagonal)

    IDs = []
    with open(rows_file, "r") as f:
        for line in f:
            IDs.append(line.strip())

    return data, IDs


def load_numpy(numpy_file, rows_file, fill_diagonal=None):
    """
    Reads and returns an numpy matrix and corresponding IDs.

    Parameters
    ----------
        numpy_file : str
            Path to a numpy matrix file.

    Returns
    -------
        data : list(dict)
            An numpy matrix.
        IDs : list(str)
            Gene IDs corresponding to each row of `data`.
    """
    data = np.load(numpy_file)

    if fill_diagonal is not None:
        np.fill_diagonal(data, fill_diagonal)

    IDs = []
    with open(rows_file, "r") as f:
        for line in f:
            IDs.append(line.strip())

    return data, IDs


def load_feature_row_file(feature_file, rows_file):
    """
    Reads a libSVM-formatted feature file and corresponding row IDs and returns
    an adjacency dictionary and IDs.

    Parameters
    ----------
        feature_file : str
            Path to a LibSVM-formatted feature file.
        row_file : str
            Path to a file containing a newline-separated list of gene IDs
            associated with each row of feature_file.

    Returns
    -------
        data : list(dict)
            A list of feature vector dicts where elements are accessed by gene
            ID.
        IDs : list(str)
            Gene IDs corresponding to each row of `data`.
    """
    labels, data = ll.svm_read_problem(feature_file)
    IDs = []
    with open(rows_file, "r") as f:
        for line in f:
            IDs.append(line.strip())

    return data, IDs


def load_labels(labels_file):
    """
    Reads a labels file and returns a list of genes and corresponding labels.

    Assumes lables are ints: 1 (positive), -1 (negative), or 0 (unlabeled).

    Parameters
    ----------
        labels_file : str
            Path to a tab-separated file with lines formatted like
            gene ID \\t label \\t weight

    Returns
    -------
        IDs : list(str)
            Gene IDs that have labels.
        labels : numpy.nd_array(int)
            Labels corresponding to `IDs`.
    """
    IDs = []
    labels = []

    with open(labels_file, "r") as f:
        f.readline()
        for line in f:
            ID, label = line.split("\t")
            IDs.append(ID)
            labels.append(int(label))

    return IDs, labels


def load_weighted_labels(labels_file):
    """
    Reads a labels file and returns a list of genes and corresponding labels
    and weights.

    Assumes lables are ints: 1 (positive), -1 (negative), or 0 (unlabeled).

    Parameters
    ----------
        labels_file : str
            Path to a tab-separated file with lines formatted like
            gene ID \\t label \\t weight

    Returns
    -------
        IDs : list(str)
            Gene IDs that have labels.
        labels : numpy.nd_array(int)
            Labels corresponding to `IDs`.
        weights : numpy.nd_array(float)
            Weights corresponding to `IDs`.
    """
    IDs = []
    labels = []
    weights = []

    with open(labels_file, "r") as f:
        for line in f:
            ID, label, weight = line.split("\t")
            IDs.append(ID)
            labels.append(int(label))
            weights.append(float(weight))

    labels = np.array(labels)
    weights = np.array(weights)

    return IDs, labels, weights


def get_class_data(data, IDs, labeled_IDs, labels, weights, classes):
    """
    Removes data rows that do not have specified class labels and returns
    resulting data matrix and the corresponding labeled gene IDs.

    Parameters
    ----------
        data : list(dict)
            List of feature dicts.
        IDs : list(str)
            IDs associated with each row of `data`.
        labeled_IDs : list(str)
            IDs from labels file.
        labels : numpy.nd_array(int)
            Labels associated with `labeled_IDs`.
        weights : numpy.nd_array(float)
            Weights associated with `labeled_IDs`.

    Returns
    -------
        class_data : list(dict)
            List of feature dicts with only data of specified class.
        new_class_IDs : list(str)
            IDs associated with each row of `class_data`.
        new_labels : numpy.nd_array(int)
            Labels associated with `new_class_IDs`.
        new_weights : numpy.nd_array(float)
            Weights associated with `new_class_IDs`.
    """
    class_indices = [
        i for i in np.where(np.isin(labels, classes))[0]
        if labeled_IDs[i] in IDs]
    class_IDs = [labeled_IDs[i] for i in class_indices]
    class_labels = labels[class_indices]
    class_weights = weights[class_indices]

    data_indices = [IDs.index(ID) for ID in class_IDs if ID in IDs]
    if type(data) == np.ndarray:
        class_data = data[data_indices]
    else:
        class_data = [data[i] for i in data_indices]

    return class_data, class_IDs, class_labels, class_weights


def load_predictions(predictions_file):
    """
    Reads a predictions file and returns a list of genes, their predicted
    labels, and the probability with which they are a member of the positive
    class.

    Parameters
    ----------
        predictions_file : str
            Path to a tab-separated file with lines formatted like
            gene ID \\t predicted label \\t probability

    Returns
    -------
        predicted_IDs : list(str)
            Gene IDs that have labels and predictions.
        predicted_labels : numpy.nd_array(int)
            Labels corresponding to `predicted_IDs`.
        probs : numpy.nd_array(float)
            Probabilities corresponding to `predicted_IDs`.
    """
    predicted_IDs = []
    probs = []

    with open(predictions_file, "r") as f:
        for line in f:
            ID, prob = line.split("\t")
            predicted_IDs.append(ID)
            probs.append(float(prob))

    probs = np.array(probs)

    return predicted_IDs, probs


def train_test_split(labels, weights, evidence_levels, k):
    """
    Creates a k-fold train-test split with all requested evidence levels.
    Negative labels are included regardless of evidence level.

    Parameters
    ----------
        labels : numpy.nd_array(int)
            Labels associated with each row of `data`. Assumes labels are 1 or
            -1.
        weights : numpy.nd_array(float)
            Weights associated with each row of `data`.
        evidence_levels : list(int) or str
            Evidence level(s) to use in training. 1 is the highest evidence
            level, followed by 2, 3, ..., n. Can also be "all" to include all
            evidence.
        k : int
            Number of folds to use in train/test splits.

    Returns
    -------
        train_indices : list(numpy.nd_array(int))
            List of length `k` containing training indices for each fold.
        test_indices : list(numpy.nd_array(int))
            List of length `k` containing testing indices for each fold.
    """
    # get list of evidence weights ordered by evidence level
    evidence_weights = np.unique(weights)[::-1]

    # get requested evidence levels
    if evidence_levels == "all":
        indices = np.arange(len(weights))
    else:
        negatives = labels == -1
        requested_weights = \
            [evidence_weights[int(E) - 1] for E in evidence_levels]

        indices = np.where(
            np.logical_or(negatives, np.in1d(weights, requested_weights)))[0]

    # split requested indices into k folds
    train_indices, test_indices = \
        list(zip(*(KFold(n_splits=k, shuffle=True).split(indices))))

    train_indices = [indices[train_indices[fold]] for fold in range(k)]
    test_indices = [indices[test_indices[fold]] for fold in range(k)]

    return train_indices, test_indices
