import pandas as pd

with open('data/Xpresso_genes.txt', 'r') as f:
    genes = f.readlines()
genes = [gene.strip() for gene in genes]

# read the targetscan data
targets = pd.read_csv("data/Predicted_Targets_Context_Scores.default_predictions.txt", sep="\t")

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

# compute scores for each gene-mirna pair
ScoreSums = targets.groupby(["miRNA", "Gene ID"]).sum()

# create a matrix with genes as rows and mirnas as columns, with corresponding scores as values
ScoreMatrix = ScoreSums.pivot_table(index="Gene ID", columns="miRNA", values="weighted context++ score")
# fill missing values with 0
ScoreMatrix = ScoreMatrix.fillna(0)
# for the genes that are not in the list, add a row of zeros
for gene in genes:
    if gene not in ScoreMatrix.index:
        ScoreMatrix.loc[gene] = [0] * ScoreMatrix.shape[1]

# save the matrix
ScoreMatrix.to_csv("data/ScoreMatrix.csv", sep="\t")