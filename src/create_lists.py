import pandas as pd
import numpy as np
import h5py
import argparse
import os

from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr


def main(args):

    df = pd.read_csv(os.path.join('figures', "results_no_mirna.csv"), sep='\t', header=[0,1], index_col=0)

    ScoreMatrix = pd.read_csv(os.path.join(args.data_dir, "ScoreMatrix.csv"), sep="\t", index_col=0)

    for primary_site in args.primary_sites:
        for cancer_subtype in args.cancer_subtypes[primary_site]:
            best_n = df[primary_site][cancer_subtype].idxmax()
            print('Loading predictions and results for: ', primary_site, cancer_subtype, best_n)
            preds = pd.read_csv(os.path.join(args.results_dir, primary_site, 'no_mirna', cancer_subtype, str(best_n), 'predictions.txt'), sep='\t', index_col=0)
            genes = list(preds.index)
            # print(len(genes))
            mirnas = h5py.File(os.path.join(args.data_dir, 'h5_datasets', f'{primary_site}.h5'), 'r')
            mirnas = mirnas['mirnaName']
            mirnas = [m.decode() for m in mirnas]
            # print(len(mirnas))

            # the genes in the score matrix may not be in the same order as in the predictions
            # therefore we need to reorder the score matrix
            scores = ScoreMatrix.loc[genes]

            # compute the adjusted predictions and use them to compute the residuals
            x = preds["Pred"].to_numpy().reshape(-1, 1)
            y = preds["Actual"].to_numpy()
            reg = LinearRegression().fit(x, y)

            predAdj = reg.predict(x)

            res = predAdj - y


            # compute the correlation between the residuals and the scores
            columns = list(scores.columns)
            cors = []
            cors_abs = []

            for col in scores:
                cors.append(spearmanr(res, scores[col])[0])
                cors_abs.append(spearmanr(abs(res), scores[col])[0])

            cors = np.array(cors)
            cors_abs = np.array(cors_abs)
            columns = np.array(columns)

            # find the index of the 10 best correlated miRNAs
            best = np.argsort(cors)[-10:]
            chosen_mirnas = columns[best]

            best_abs = np.argsort(cors_abs)[-10:]
            chosen_mirnas_abs = columns[best_abs]

            with open(os.path.join(args.output_dir, f"corr_{primary_site}_{cancer_subtype}.txt"), "w") as f:
                for m in sorted(chosen_mirnas):
                    f.write(m)
                    f.write("\n")

            with open(os.path.join(args.output_dir, f"corr_abs_{primary_site}_{cancer_subtype}.txt"), "w") as f:
                for m in sorted(chosen_mirnas_abs):
                    f.write(m)
                    f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--primary_sites", type=list, default=['lung', 'breast', 'kidney', 'uterus'])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="mirna_lists")

    args = parser.parse_args()

    cancer_subtypes = OrderedDict()
    cancer_subtypes = {primary_site: os.listdir(os.path.join(args.results_dir, primary_site, 'no_mirna')) for primary_site in args.primary_sites}
    args.cancer_subtypes = cancer_subtypes
    

    # check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)