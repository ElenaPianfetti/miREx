"""Processes the TargetScan data to obtain a matrix of scores for each gene-mirna pair, saves the targetscan data of interest"""
import pandas as pd
import argparse

def process_targetscan(targetscan_file, genes):
    # read the targetscan data
    print('Reading TargetScan data...')
    targets = pd.read_csv(args.targetscan_file, sep="\t")

    print('Processing TargetScan data...')
    # consider only human targets
    targets = targets[targets["Gene Tax ID"] == 9606]
    # the columns of interest are Gene ID, miRNA, weighted context++ score
    targets = targets[["Gene ID", "miRNA", "weighted context++ score"]]
    # remove the version of the gene
    target_genes = list(targets["Gene ID"])
    target_genes = [gene.split(".")[0] for gene in target_genes]
    targets["Gene ID"] = target_genes

    # rename mirnas so that they are all in the same format as the TCGA data
    mirs = list(targets["miRNA"])
    # lower case
    mirs = [m.lower() for m in mirs]
    mirs = [m.replace(".", "-") for m in mirs]
    # there is no arm information in the TCGA data
    mirs = [m.replace("-3p", "") for m in mirs]
    mirs = [m.replace("-5p", "") for m in mirs]
    targets["miRNA"] = mirs

    # keep only the genes in the list
    targets = targets[targets["Gene ID"].isin(genes)]

    return targets


def create_matrices(targets, genes):
    print('Summing scores...')
    # compute scores for each gene-mirna pair
    ScoreSums = targets.groupby(["miRNA", "Gene ID"]).sum()
    print('Creating matrices...')
    # create a matrix with genes as rows and mirnas as columns, with corresponding scores as values
    ScoreMatrix = ScoreSums.pivot_table(index="Gene ID", columns="miRNA", values="weighted context++ score")
    # fill missing values with 0
    ScoreMatrix = ScoreMatrix.fillna(0)
    # for the genes that are not in the list, add a row of zeros
    for gene in genes:
        if gene not in ScoreMatrix.index:
            ScoreMatrix.loc[gene] = [0] * ScoreMatrix.shape[1]
    print(ScoreMatrix.shape)

    # save the matrix
    ScoreMatrix.to_csv("data/ScoreMatrix.csv", sep="\t")

    # create a matrix where the values are 1 if mirna targets the gene, 0 otherwise
    TargetMatrix = ScoreMatrix.copy()
    TargetMatrix[TargetMatrix != 0] = 1
    TargetMatrix.to_csv("data/TargetMatrix.csv", sep="\t")

def main(args):
    with open(args.genes_file, 'r') as f:
        genes = f.readlines()
    genes = [gene.strip() for gene in genes]

    targets = process_targetscan(args.targetscan_file, genes)

    create_matrices(targets, genes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--targetscan_file', type=str, default='data/Predicted_Targets_Context_Scores.default_predictions.txt', help='TargetScan file')
    parser.add_argument('--genes_file', type=str, default='data/Xpresso_genes.txt', help='File with the list of genes')
    args = parser.parse_args()
    main(args)