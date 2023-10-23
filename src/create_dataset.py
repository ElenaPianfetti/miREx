"""Creates h5 datasets from the raw data."""

import argparse
import os
import sys
import h5py
import pandas as pd
import json
from tqdm import tqdm
import numpy as np

from utils.utils import parse_strings

def create_csvs(args):
    for primary_site in args.primary_sites:
        for omic in args.omics:
            if os.path.exists(os.path.join(args.csv_dir, f"{primary_site}_{omic}.csv")):
                print(f"File {primary_site}_{omic}.csv already exists")
                continue
            # add expression values for all samples in one dataframe
            df_total = pd.DataFrame()
            with open(os.path.join(args.data_dir, primary_site, f"{primary_site}_{omic}.json"), 'r') as f:
                data_dict = json.load(f)

            print(f"Creating {primary_site}_{omic}.csv")
            # read all the files
            dirs = os.listdir(os.path.join(args.data_dir, primary_site, omic))
            for d in tqdm(dirs):
                files = os.listdir(os.path.join(args.data_dir, primary_site, omic, d))
                files = [file for file in files if 'annotations' not in file and 'logs' not in file]
                # find element in the json with the same name as the file
                element = [element for element in data_dict if element['file_name'] == files[0]]
                cl_name = element[0]['cases'][0]['project']['project_id'].split('-')[1]
                if len(files) != 1:
                    print(f"Error: {len(files)} files found in {d}")
                    sys.exit()
                if omic == 'mRNA':
                    df = pd.read_csv(os.path.join(args.data_dir, primary_site, omic, d, files[0]), skiprows=[2, 3, 4, 5], sep='\t', header=1, index_col=0, usecols=['gene_id', 'unstranded'])
                    exp_genes = df.index.values
                    exp_genes = [gene for gene in exp_genes if 'PAR_Y' not in gene]
                    df = df.loc[exp_genes]
                    df.index = df.index.str.split('.').str[0]
                    df.columns = [cl_name]
                    df_total = pd.concat([df_total, df], axis=1)
                elif omic == 'miRNA':
                    df = pd.read_csv(os.path.join(args.data_dir, primary_site, omic, d, files[0]), sep='\t', usecols=['miRNA_ID', 'read_count'], index_col=0)
                    df.columns = [cl_name]
                    df_total = pd.concat([df_total, df], axis=1)
            n_classes = len(set(df_total.columns.values))
            classes = list(set(df_total.columns.values))
            # count the number of times each class appears
            class_counts = df_total.columns.value_counts()
            # exclude classes with low number of samples
            min_samples = 10
            excluded_classes = class_counts[class_counts <= min_samples].index.values
            print(f"{len(excluded_classes)} classes with less than {min_samples} samples: {excluded_classes}")
            classes = class_counts[class_counts > min_samples].index.values
            n_classes = len(classes)
            df_total = df_total[classes]
            df_total.to_csv(os.path.join(args.csv_dir, f"{primary_site}_{omic}_complete.csv"), index=True, header=True, sep='\t')
            print(f"Number of classes: {n_classes}: {classes}")
            print(f"Number of samples: {df_total.shape[1]}")
            for cl in classes:
                print(f"{cl}: {class_counts[cl]}")

            df_mean = pd.DataFrame()
            # compute the mean and normalize the data
            for cl in classes:
                df_mean[cl] = np.log10(df_total[cl].mean(axis=1) + 0.1)
            if n_classes > 1:
                df_mean['mean'] = np.log10(df_total.mean(axis=1) + 0.1)
            
            df_mean.to_csv(os.path.join(args.csv_dir, f"{primary_site}_{omic}.csv"), index=True, header=True, sep='\t')

def filter_csvs(args):
    targets = pd.read_csv(args.targetscan_file, sep="\t", index_col=0)
    genes = list(targets.index.values)
    miRNA_targets = list(targets.columns.values)

    for primary_site in args.primary_sites:
        for omic in args.omics:
            print(f"Filtering {primary_site}_{omic}.csv")
            df = pd.read_csv(os.path.join(args.csv_dir, f"{primary_site}_{omic}.csv"), sep='\t', index_col=0)
            if omic == 'mRNA':
                df = df[df.index.isin(genes)]
            elif omic == 'miRNA':
                df = df[df.index.isin(miRNA_targets)]
            df.to_csv(os.path.join(args.csv_dir, f"{primary_site}_{omic}_filtered.csv"), index=True, header=True, sep='\t')

    df_total = pd.read_csv(os.path.join(args.csv_dir, f"{primary_site}_mRNA_complete.csv"), sep='\t', index_col=0, header=0)
    df_total.columns = [col.split('.')[0] for col in df_total.columns]
    df_total = df_total[df_total.index.isin(genes)]

    df_total.to_csv(os.path.join(args.csv_dir, f"{primary_site}_mRNA_complete_filtered.csv"), index=True, header=True, sep='\t')

def create_h5(args):

    for primary_site in args.primary_sites:
        print(f"\nCreating {primary_site} datasets")
        dataset = h5py.File(os.path.join(args.h5_dir, f"{primary_site}.h5"), 'w')

        TargetMatrix = pd.read_csv(os.path.join(args.targetscan_file), sep="\t", index_col=0)
        
        df_mRNA = pd.read_csv(os.path.join(args.csv_dir, f"{primary_site}_mRNA_filtered.csv"), sep='\t', index_col=0)
        genes = list(df_mRNA.index.values)
        print(f"Number of genes:\t{len(genes)}")
        # count how many times each gene appears
        gene_counts = df_mRNA.index.value_counts()
        df_miRNA = pd.read_csv(os.path.join(args.csv_dir, f"{primary_site}_miRNA_filtered.csv"), sep='\t', index_col=0)
        mirnas = list(df_miRNA.index.values)
        print(f"Number of miRNAs:\t{len(mirnas)}")
        n_classes = len(df_mRNA.columns.values)
        classes = list(df_mRNA.columns.values)
        print(f"{n_classes} classes:\t\t{classes}")

        dataset.create_dataset('geneName', data=np.char.encode(genes))
        dataset.create_dataset('mirnaName', data=np.char.encode(mirnas))

        # the keys should be in the form of geneName, <class_name>_mRNA, miRNAName, <class_name>_miRNA
        # for the mRNAs, the csvs columns are enough, for the miRNAs, we have to create the matrices for each class
        # the matrices will be of shape (n_genes, n_miRNAs)
        TargetMatrix = TargetMatrix.loc[genes, mirnas]
        for cl in classes:
            cl_df = TargetMatrix.copy()
            for miRNA in mirnas:
                expression = df_miRNA.loc[miRNA, cl]
                cl_df[miRNA] = cl_df[miRNA] * expression
            dataset.create_dataset(f"{cl}_mRNA", data=df_mRNA[cl].values)
            dataset.create_dataset(f"{cl}_miRNA", data=cl_df.values)

def main(args):
    # create the csvs with the mean for each class + normalize the data
    create_csvs(args)
    # filter the csvs to only include the genes in the Xpresso dataset, and the miRNA targets
    filter_csvs(args)
    # create the h5 files
    create_h5(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/gdc_data", help="Directory with the data")
    parser.add_argument("--h5_dir", type=str, default="data/h5_datasets", help="Directory with the h5 datasets")
    parser.add_argument("--csv_dir", type=str, default="data/csvs", help="Directory with the csv files")
    parser.add_argument('--targetscan_file', type=str, default='data/TargetMatrix.csv', help='TargetScan file')
    parser.add_argument("--primary_sites", type=parse_strings, default=['lung', 'breast', 'kidney', 'uterus'], help="Cancer types to use")
    parser.add_argument("--omics", type=parse_strings, default=['mRNA', 'miRNA'], help="Omics to use")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    if args.verbose:
        print('\nStarting with arguments:')
        # get longest argument name
        max_len = max([len(arg) for arg in vars(args)])
        for arg in vars(args):
            print(f"{arg:{max_len}} : {getattr(args, arg)}")
        print('\n')
    main(args)