import pandas as pd
import numpy as np
import h5py
import argparse
import os

from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/no_mirna")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="mirna_lists")
    parser.add_argument("--best_mean", type=str, default="mean/3/predictions.txt")
    parser.add_argument("--best_LUAD", type=str, default="LUAD/2/predictions.txt")
    parser.add_argument("--best_LUSC", type=str, default="LUSC/5/predictions.txt")

    args = parser.parse_args()

    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mean = pd.read_csv(os.path.join(args.results_dir, args.best_mean), sep="\t", index_col=0)
    LUAD = pd.read_csv(os.path.join(args.results_dir, args.best_LUAD), sep="\t", index_col=0)
    LUSC = pd.read_csv(os.path.join(args.results_dir, args.best_LUSC), sep="\t", index_col=0)

    # get the list of genes
    genes = list(mean.index)

    names_mir = h5py.File(os.path.join(args.data_dir, "h5_datasets/train.h5"), "r")
    mean_mir = names_mir["mirnaName_mean"]
    LUAD_mir = names_mir["mirnaName_LUAD"]
    LUSC_mir = names_mir["mirnaName_LUSC"]
    mean_mir = [m.decode() for m in mean_mir]
    LUAD_mir = [m.decode() for m in LUAD_mir]
    LUSC_mir = [m.decode() for m in LUSC_mir]


    scores = pd.read_csv(os.path.join(args.data_dir, "ScoreMatrix.csv"), sep="\t", index_col=0)

    # the genes in the score matrix are not in the same order as in the predictions
    # therefore we need to reorder the score matrix
    scores = scores.loc[genes]
    scores_mean = scores[mean_mir]
    scores_LUAD = scores[LUAD_mir]
    scores_LUSC = scores[LUSC_mir]



    # compute the adjusted predictions and use them to compute the residuals
    x_mean = mean["Pred"].to_numpy().reshape(-1, 1)
    x_LUAD = LUAD["Pred"].to_numpy().reshape(-1, 1)
    x_LUSC = LUSC["Pred"].to_numpy().reshape(-1, 1)
    y_mean = mean["Actual"].to_numpy()
    y_LUAD = LUAD["Actual"].to_numpy()
    y_LUSC = LUSC["Actual"].to_numpy()
    reg_mean = LinearRegression().fit(x_mean, y_mean)
    reg_LUSC = LinearRegression().fit(x_LUSC, y_LUSC)
    reg_LUAD = LinearRegression().fit(x_LUAD, y_LUAD)

    predAdj_mean = reg_mean.predict(x_mean)
    predAdj_LUSC = reg_LUSC.predict(x_LUSC)
    predAdj_LUAD = reg_LUAD.predict(x_LUAD)

    res_mean = predAdj_mean - y_mean
    res_LUAD = predAdj_LUAD - y_LUAD
    res_LUSC = predAdj_LUSC - y_LUSC


    # compute the correlation between the residuals and the scores
    columns_mean = list(scores_mean.columns)
    columns_LUAD = list(scores_LUAD.columns)
    columns_LUSC = list(scores_LUSC.columns)
    cors_mean = []
    cors_LUAD = []
    cors_LUSC = []
    cors_mean_abs = []
    cors_LUAD_abs = []
    cors_LUSC_abs = []


    for col in scores_mean:
        cors_mean.append(spearmanr(res_mean, scores_mean[col])[0])
        cors_mean_abs.append(spearmanr(abs(res_mean), scores_mean[col])[0])

    for col in scores_LUAD:
        cors_LUAD.append(spearmanr(res_LUAD, scores_LUAD[col])[0])
        cors_LUAD_abs.append(spearmanr(abs(res_LUAD), scores_LUAD[col])[0])

    for col in scores_LUSC:
        cors_LUSC.append(spearmanr(res_LUSC, scores_LUSC[col])[0])
        cors_LUSC_abs.append(spearmanr(abs(res_LUSC), scores_LUSC[col])[0])

    cors_mean = np.array(cors_mean)
    cors_LUAD = np.array(cors_LUAD)
    cors_LUSC = np.array(cors_LUSC)
    cors_mean_abs = np.array(cors_mean_abs)
    cors_LUAD_abs = np.array(cors_LUAD_abs)
    cors_LUSC_abs = np.array(cors_LUSC_abs)

    columns_mean = np.array(columns_mean)
    columns_LUAD = np.array(columns_LUAD)
    columns_LUSC = np.array(columns_LUSC)

    # find the index of the 10 best correlated miRNAs
    best_mean = np.argsort(cors_mean)[-10:]
    chosen_mirnas_mean = columns_mean[best_mean]

    best_LUAD = np.argsort(cors_LUAD)[-10:]
    chosen_mirnas_LUAD = columns_LUAD[best_LUAD]

    best_LUSC = np.argsort(cors_LUSC)[-10:]
    chosen_mirnas_LUSC = columns_LUSC[best_LUSC]

    best_mean_abs = np.argsort(cors_mean_abs)[-10:]
    chosen_mirnas_mean_abs = columns_mean[best_mean_abs]

    best_LUAD_abs = np.argsort(cors_LUAD_abs)[-10:]
    chosen_mirnas_LUAD_abs = columns_LUAD[best_LUAD_abs]

    best_LUSC_abs = np.argsort(cors_LUSC_abs)[-10:]
    chosen_mirnas_LUSC_abs = columns_LUSC[best_LUSC_abs]

    with open(os.path.join(args.output_dir, "corr_mean.txt"), "w") as f:
        for m in chosen_mirnas_mean:
            f.write(m)
            f.write("\n")
    with open(os.path.join(args.output_dir, "corr_LUAD.txt"), "w") as f:
        for m in chosen_mirnas_LUAD:
            f.write(m)
            f.write("\n")
    with open(os.path.join(args.output_dir, "corr_LUSC.txt"), "w") as f:
        for m in chosen_mirnas_LUSC:
            f.write(m)
            f.write("\n")
    with open(os.path.join(args.output_dir, "corr_abs_mean.txt"), "w") as f:
        for m in chosen_mirnas_mean_abs:
            f.write(m)
            f.write("\n")
    with open(os.path.join(args.output_dir, "corr_abs_LUAD.txt"), "w") as f:
        for m in chosen_mirnas_LUAD_abs:
            f.write(m)
            f.write("\n")
    with open(os.path.join(args.output_dir, "corr_abs_LUSC.txt"), "w") as f:
        for m in chosen_mirnas_LUSC_abs:
            f.write(m)
            f.write("\n")
