#!/bin/bash
DIR="$( dirname "$(readlink -f "$0")" )"/..
export PYTHONPATH=${PYTHONPATH}:${DIR}/bin/liblinear-weights-2.30/python

n_trials=10
n_folds=5

rm -rf ${DIR}/results
mkdir -p ${DIR}/results/splits
mkdir -p ${DIR}/results/predictions
mkdir -p ${DIR}/results/unsplits
mkdir -p ${DIR}/results/evaluations
mkdir -p ${DIR}/results/plots

echo 'Splitting labels into cross-validation folds...'
python ${DIR}/src/split.py --labels ${DIR}/data/labels/E*.tsv ${DIR}/data/labels/negatives.tsv --n_trials ${n_trials} --k_folds ${n_folds} --output ${DIR}/results/splits

echo 'Training models & predicting on test folds...'

for trial in $(seq 1 $n_trials); do
    echo "Trial ${trial}:"
    for fold in $(seq 1 $n_folds); do
        echo "Fold ${fold}:"

        echo "Training on E1 only..."
        python ${DIR}/src/predict.py --numpy ${DIR}/data/networks/brain.npy --rows ${DIR}/data/networks/brain_genes.tsv --fill_diagonal 1 --labels ${DIR}/results/splits/negatives_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E1_trial${trial}_fold${fold}.tsv --output ${DIR}/results/predictions/E1

        echo "Training on E1 + E2..."
        python ${DIR}/src/predict.py --numpy ${DIR}/data/networks/brain.npy --rows ${DIR}/data/networks/brain_genes.tsv --fill_diagonal 1 --labels ${DIR}/results/splits/negatives_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E1_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E2_trial${trial}_fold${fold}.tsv --output ${DIR}/results/predictions/E1+E2

        echo "Training on E1 + E2 + E3 + E4..."
        python ${DIR}/src/predict.py --numpy ${DIR}/data/networks/brain.npy --rows ${DIR}/data/networks/brain_genes.tsv --fill_diagonal 1 --labels ${DIR}/results/splits/negatives_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E*_trial${trial}_fold${fold}.tsv --output ${DIR}/results/predictions/E1+E2+E3+E4
    done
done

echo 'Evaluating predictions...'
for trial in $(seq 1 $n_trials); do
    echo "Trial ${trial}:"
    for fold in $(seq 1 $n_folds); do
        echo "Fold ${fold}:"

        echo "Evaluating E1 only..."
        python ${DIR}/src/unsplit.py --splits ${DIR}/results/splits/negatives_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E1_trial${trial}_fold${fold}.tsv --predictions ${DIR}/results/predictions/E1/trial${trial}_fold${fold}.tsv --output ${DIR}/results/unsplits/E1

        python ${DIR}/src/evaluate.py --labels ${DIR}/data/labels/E1.tsv ${DIR}/data/labels/negatives.tsv --predictions ${DIR}/results/unsplits/E1/trial${trial}_fold${fold}.tsv --output ${DIR}/results/evaluations/E1

        echo "Evaluating E1 + E2..."
        python ${DIR}/src/unsplit.py --splits ${DIR}/results/splits/negatives_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E1_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E2_trial${trial}_fold${fold}.tsv --predictions ${DIR}/results/predictions/E1+E2/trial${trial}_fold${fold}.tsv --output ${DIR}/results/unsplits/E1+E2

        python ${DIR}/src/evaluate.py --labels ${DIR}/data/labels/E1.tsv ${DIR}/data/labels/negatives.tsv --predictions ${DIR}/results/unsplits/E1+E2/trial${trial}_fold${fold}.tsv --output ${DIR}/results/evaluations/E1+E2

        echo "Evaluating E1 + E2 + E3 + E4..."
        python ${DIR}/src/unsplit.py --splits ${DIR}/results/splits/negatives_trial${trial}_fold${fold}.tsv ${DIR}/results/splits/E*_trial${trial}_fold${fold}.tsv --predictions ${DIR}/results/predictions/E1+E2+E3+E4/trial${trial}_fold${fold}.tsv --output ${DIR}/results/unsplits/E1+E2+E3+E4

        python ${DIR}/src/evaluate.py --labels ${DIR}/data/labels/E1.tsv ${DIR}/data/labels/negatives.tsv --predictions ${DIR}/results/unsplits/E1+E2+E3+E4/trial${trial}_fold${fold}.tsv --output ${DIR}/results/evaluations/E1+E2+E3+E4
    done
done

echo 'Generating plots...'
python ${DIR}/src/plot.py --results ${DIR}/results --evaluations E1+E2+E3+E4 E1+E2 E1


