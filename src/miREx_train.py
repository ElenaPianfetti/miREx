# set up imports
import h5py
import os
from scipy import stats
import numpy as np
import pandas as pd
import sys
import argparse

from utils.utils import parse_and_check

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def main(args):
    for i in range(args.n_train):
        print(f'Training #{i+1}')
        tr = 0
        while tr < 0.01:
            load_data(args, i)
            # ------------------- params -------------------
            params = {
                "mir": args.mir,
                "resultsdir": os.path.join(args.resultsdir, str(i)),
                "batchsize": 2 ** 5,
                "leftpos": 3000,
                "rightpos": 13500,
                "activationFxn": "relu",
                "numFiltersConv1": 2 ** 7,
                "filterLenConv1": 6,
                "dilRate1": 1,
                "maxPool1": 30,
                "numFiltersConv2": 2 ** 5,
                "filterLenConv2": 9,
                "dilRate2": 1,
                "maxPool2": 10,
                "dense1": 2 ** 6,
                "dropout1": 0.00099,
                "dense2": 2, 
                "dropout2": 0.01546,
                "n_epochs": 100,
                "patience": 20
            }
            results = train_model(params)
            print(results)
            tr = results["test_R2"]

def load_data(args, n_train):

    global train_halflife, valid_halflife, test_halflife
    global train_promoter, valid_promoter, test_promoter
    global train_expression, valid_expression, test_expression
    global train_miRNA, valid_miRNA, test_miRNA
    global train_geneName, valid_geneName, test_geneName

    # ------------------- load data -------------------

    if args.old_dataset:
        old_data_dir = '/work/H2020DeciderFicarra/epianfetti/Xpresso_polmoneGDC/data/h5_datasets'
        train_file = h5py.File(os.path.join(old_data_dir, "train.h5"), "r")
        test_file = h5py.File(os.path.join(old_data_dir, "test.h5"), "r")
        valid_file = h5py.File(os.path.join(old_data_dir, "valid.h5"), "r")

        train_halflife = train_file["data"][:]
        test_halflife = test_file["data"][:]
        valid_halflife = valid_file["data"][:]

        train_promoter = train_file["promoter"][:]
        test_promoter = test_file["promoter"][:]
        valid_promoter = valid_file["promoter"][:]

        train_expression = train_file[f"{args.cancer_subtype}_no_FPKM"][:]
        test_expression = test_file[f"{args.cancer_subtype}_no_FPKM"][:]
        valid_expression = valid_file[f"{args.cancer_subtype}_no_FPKM"][:]

        train_miRNA = train_file[f"{args.cancer_subtype}_mirna"][:]
        test_miRNA = test_file[f"{args.cancer_subtype}_mirna"][:]
        valid_miRNA = valid_file[f"{args.cancer_subtype}_mirna"][:]

        train_geneName = train_file["geneName"][:]
        test_geneName = test_file["geneName"][:]
        valid_geneName = valid_file["geneName"][:]
    else:

        sequence_data_file = h5py.File(os.path.join(args.datadir, "sequence_data.h5"), "r")
        halflife = sequence_data_file["halflife"][:]
        promoter = sequence_data_file["promoter"][:]
        genes = sequence_data_file["geneName"][:]
        genes = [gene.decode() for gene in genes]
        sequence_data_file.close()

        primary_site_file = h5py.File(os.path.join(args.datadir, f"{args.primary_site}.h5"), "r")
        # find indices of miRNAs in the dataset

        expression = primary_site_file[f"{args.cancer_subtype}_mRNA"][:]
        geneName = primary_site_file["geneName"][:]
        geneName = [gene.decode() for gene in geneName]

        mirNames = primary_site_file[f"mirnaName"]
        mirNames = [mi.decode() for mi in mirNames]
        if args.mirna_list:
            mirna_list = list(set(args.mirna_list).intersection(set(mirNames)))
            mir_indeces = [mirNames.index(mi) for mi in mirna_list]
        else:
            mir_indeces = [True] * len(mirNames)
        miRNA = primary_site_file[f"{args.cancer_subtype}_miRNA"][:]
        miRNA = miRNA[:, mir_indeces]
        primary_site_file.close()

        # the sequence data has more genes than the target data, find the indices of the genes that are in both
        genes_indeces = [genes.index(gene) for gene in geneName if gene in genes]
        promoter = promoter[genes_indeces, :, :]
        halflife = halflife[genes_indeces, :]
        genes = [genes[i] for i in genes_indeces]

        if args.verbose:
            # print the shapes of the data
            print("halflife shape:\t\t", halflife.shape)
            print("promoter shape:\t\t", promoter.shape)
            print("expression shape:\t", expression.shape)
            print("miRNA shape:\t\t", miRNA.shape)
            print("geneName shape:\t\t", len(geneName))
            print("miRNAName shape:\t", len(mirNames))
            print("")


        if args.Xpresso_split:
            with open(os.path.join(args.genes_dir, 'train_genes.txt')) as train_file, open(os.path.join(args.genes_dir, 'test_genes.txt')) as test_file, open(os.path.join(args.genes_dir, 'valid_genes.txt')) as valid_file:
                train_genes = train_file.readlines()
                test_genes = test_file.readlines()
                valid_genes = valid_file.readlines()
                train_genes = [gene.strip() for gene in train_genes]
                test_genes = [gene.strip() for gene in test_genes]
                valid_genes = [gene.strip() for gene in valid_genes]
                train_genes = list(set(train_genes).intersection(set(genes)))
                test_genes = list(set(test_genes).intersection(set(genes)))
                valid_genes = list(set(valid_genes).intersection(set(genes)))

                training_indices = [genes.index(gene) for gene in train_genes if gene in genes]
                test_indices = [genes.index(gene) for gene in test_genes if gene in genes]
                validation_indices = [genes.index(gene) for gene in valid_genes if gene in genes]
        else:
            raise NotImplementedError
            # ------------------- split data -------------------
            # 1000 genes in test set, 1000 genes in validation set, rest in training set
            test_size = 1000
            validation_size = 1000
            total_elements = len(genes)
            indices = np.arange(total_elements)

            np.random.shuffle(indices)

            test_indices = indices[:test_size]
            validation_indices = indices[test_size:test_size+validation_size]
            training_indices = indices[test_size+validation_size:]

        # split each dataset into train, test, and validation sets
        # ------------------- halflife -------------------
        train_halflife = halflife[training_indices, :]
        test_halflife = halflife[test_indices, :]
        valid_halflife = halflife[validation_indices, :]
        # ------------------- promoter -------------------
        train_promoter = promoter[training_indices, :, :]
        test_promoter = promoter[test_indices, :, :]
        valid_promoter = promoter[validation_indices, :, :]
        # ------------------- expression -------------------
        train_expression = expression[training_indices]
        test_expression = expression[test_indices]
        valid_expression = expression[validation_indices]
        # ------------------- miRNA -------------------
        train_miRNA = miRNA[training_indices, :]
        test_miRNA = miRNA[test_indices, :]
        valid_miRNA = miRNA[validation_indices, :]
        # ------------------- geneName -------------------
        train_geneName = [genes[i] for i in training_indices]
        test_geneName = [genes[i] for i in test_indices]
        valid_geneName = [genes[i] for i in validation_indices]

        with open(os.path.join(args.resultsdir, str(n_train), 'split', 'train_genes.txt'), 'w') as train_file, open(os.path.join(args.resultsdir, str(n_train), 'split', 'test_genes.txt'), 'w') as test_file, open(os.path.join(args.resultsdir, str(n_train), 'split', 'valid_genes.txt'), 'w') as valid_file:
            for gene in train_geneName:
                train_file.write(gene + '\n')
            for gene in test_geneName:
                test_file.write(gene + '\n')
            for gene in valid_geneName:
                valid_file.write(gene + '\n')
    
    if args.verbose:
        # print the shapes of the data
        print("train halflife shape:\t", train_halflife.shape) 
        print("train promoter shape:\t", train_promoter.shape)
        print("train expression shape:\t", train_expression.shape)
        print("train miRNA shape:\t", train_miRNA.shape)
        print("train geneName shape:\t", len(train_geneName))
        print("test halflife shape:\t", test_halflife.shape)
        print("test promoter shape:\t", test_promoter.shape)
        print("test expression shape:\t", test_expression.shape)
        print("test miRNA shape:\t", test_miRNA.shape)
        print("test geneName shape:\t", len(test_geneName))
        print("valid halflife shape:\t", valid_halflife.shape)
        print("valid promoter shape:\t", valid_promoter.shape)
        print("valid expression shape:\t", valid_expression.shape)
        print("valid miRNA shape:\t", valid_miRNA.shape)
        print("valid geneName shape:\t", len(valid_geneName))
        print("")


def train_model(params):

    leftpos = int(params["leftpos"])
    rightpos = int(params["rightpos"])
    activationFxn = params["activationFxn"]
    n_epochs = int(params["n_epochs"])
    patience = int(params["patience"])

    train_promoterSubseq = train_promoter[:, leftpos:rightpos, :]
    valid_promoterSubseq = valid_promoter[:, leftpos:rightpos, :]
    test_promoterSubseq = test_promoter[:, leftpos:rightpos, :]

    input_halflife = Input(shape=(train_halflife.shape[1:]), name="halflife")
    input_promoter = Input(shape=train_promoterSubseq.shape[1:], name="promoter")
    input_miRNA = Input(shape=train_miRNA.shape[1:], name="mirna")
    
    # ------------------- model -------------------
    x = Conv1D(
        int(params["numFiltersConv1"]),
        int(params["filterLenConv1"]),
        dilation_rate=int(params["dilRate1"]),
        padding="same",
        kernel_initializer="glorot_normal",
        input_shape=train_promoterSubseq.shape[1:],
        activation=activationFxn,
    )(input_promoter)
    x = MaxPooling1D(int(params["maxPool1"]), padding="same")(x)

    maxPool2 = int(params["maxPool2"])
    x = Conv1D(
        int(params["numFiltersConv2"]),
        int(params["filterLenConv2"]),
        dilation_rate=int(params["dilRate2"]),
        padding="same",
        kernel_initializer="glorot_normal",
        activation=activationFxn,
    )(x)
    x = MaxPooling1D(maxPool2, padding="same")(x)
    x = Flatten()(x)

    if params["mir"]:
        x = Concatenate()([x, input_halflife, input_miRNA])
    else:
        x = Concatenate()([x, input_halflife])

    x = Dense(int(params["dense1"]))(x)
    x = Activation(activationFxn)(x)
    x = Dropout(params["dropout1"])(x)
    x = Dense(int(params["dense2"]))(x)
    x = Activation(activationFxn)(x)
    x = Dropout(params["dropout2"])(x)
    main_output = Dense(1)(x)
    if params["mir"]:
        model = Model(inputs=[input_promoter, input_halflife, input_miRNA], outputs=[main_output])
    else:
        model = Model(inputs=[input_promoter, input_halflife], outputs=[main_output])

  
    model.compile(SGD(lr=0.0005, momentum=0.9), "mean_squared_error", metrics=["mean_squared_error"])
    if args.verbose:
        print(model.summary())

    modelfile = os.path.join(params["resultsdir"], "plotted_model.png")
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=modelfile)

    check_cb = ModelCheckpoint(os.path.join(params["resultsdir"], "bestparams.h5"), 
    monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=patience, verbose=1, mode="min")
    if params["mir"]:
        result = model.fit([train_promoterSubseq, train_halflife, train_miRNA], train_expression,
            batch_size=int(params["batchsize"]),
            shuffle="batch",
            epochs=n_epochs,
            validation_data=[[valid_promoterSubseq, valid_halflife, valid_miRNA], valid_expression],
            callbacks=[check_cb, earlystop_cb])

    else:
        result = model.fit([train_promoterSubseq, train_halflife], train_expression,
            batch_size=int(params["batchsize"]),
            shuffle="batch",
            epochs=n_epochs,
            validation_data=[[valid_promoterSubseq, valid_halflife], valid_expression],
            callbacks=[check_cb, earlystop_cb])
    mse_history = result.history["val_mean_squared_error"]
    mse = min(mse_history)

    # evaluate performance on test set using best learned model
    best_file = os.path.join(params["resultsdir"], "bestparams.h5")
    model = load_model(best_file)

    with open(os.path.join(params["resultsdir"], "results.txt"), "w") as f:
        if params["mir"]:
            predictions_test = model.predict([test_promoterSubseq, test_halflife, test_miRNA], batch_size=64).flatten()
            predictions_train = model.predict([train_promoterSubseq, train_halflife, train_miRNA], batch_size=64).flatten()
            predictions_valid = model.predict([valid_promoterSubseq, valid_halflife, valid_miRNA], batch_size=64).flatten()
        else:
            predictions_test = model.predict([test_promoterSubseq, test_halflife], batch_size=64).flatten()
            predictions_train = model.predict([train_promoterSubseq, train_halflife], batch_size=64).flatten()
            predictions_valid = model.predict([valid_promoterSubseq, valid_halflife], batch_size=64).flatten()
        
        _, _, r_value_te, _, _ = stats.linregress(predictions_test, test_expression)
        _, _, r_value_tr, _, _ = stats.linregress(predictions_train, train_expression)
        _, _, r_value_val, _, _ = stats.linregress(predictions_valid, valid_expression)
        
        # save results to file
        f.write("Test R^2\t")
        f.write(str(r_value_te ** 2))
        f.write("\n")
        
        f.write("Train R^2\t")
        f.write(str(r_value_tr ** 2))
        f.write("\n")
        
        f.write("Valid R^2\t")
        f.write(str(r_value_val ** 2))
        f.write("\n")

        # save predictions to file
        df = pd.DataFrame(columns=["Gene", "Pred", "Actual"])
        df["Gene"] = np.concatenate((train_geneName, test_geneName, valid_geneName))
        df["Pred"] = np.concatenate((predictions_train, predictions_test, predictions_valid))
        df["Actual"] = np.concatenate((train_expression, test_expression, valid_expression))
        df["Residual"] = df["Actual"] - df["Pred"]
        df.to_csv(os.path.join(params["resultsdir"], "predictions.txt"), index=False, header=True, sep="\t")

    return {"loss": mse, "test_R2": r_value_te ** 2}



if __name__ == "__main__":
    args = parse_and_check()
    main(args)
    

