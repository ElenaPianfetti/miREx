# set up imports
import h5py
import os
from scipy import stats
import numpy as np
import pandas as pd
import sys
import argparse

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from hyperopt import STATUS_OK


print("TF version", tf.__version__)
print("Keras version", keras.__version__)
device_name = tf.test.gpu_device_name()
print(device_name)
print("Found GPU at: {}".format(device_name))


global X_trainhalflife, X_trainpromoter, y_train, X_validhalflife, X_validpromoter, y_valid
global X_testhalflife, X_testpromoter, y_test, params, best_file
global geneName_test, geneName_valid, geneName_train
global miRNA_train, miRNA_test, miRNA_valid


def main(args, chosen_m):
    global X_trainhalflife, X_trainpromoter, y_train, X_validhalflife, X_validpromoter, y_valid
    global X_testhalflife, X_testpromoter, y_test, params
    global geneName_test, geneName_valid, geneName_train
    global miRNA_train, miRNA_valid, miRNA_test

    trainfile_xpresso = h5py.File(os.path.join(args.datadir, "train_xpresso.h5"), "r")

    # load Xpresso data
    X_trainhalflife = trainfile_xpresso["data"][:]
    X_trainpromoter = trainfile_xpresso["promoter"][:]
    
    validfile_xpresso = h5py.File(os.path.join(args.datadir, "valid_xpresso.h5"), "r")
    X_validhalflife = validfile_xpresso["data"][:]
    X_validpromoter = validfile_xpresso["promoter"][:]

    testfile_xpresso = h5py.File(os.path.join(args.datadir, "test_xpresso.h5"), "r")
    X_testhalflife = testfile_xpresso["data"][:]
    X_testpromoter = testfile_xpresso["promoter"][:]

    if args.verbose:
        print('Loaded Xpresso dataset')

    # load target data
    trainfile = h5py.File(os.path.join(args.datadir, "train.h5"), "r")

    # find indices of miRNAs in the dataset
    mirNames = trainfile[f"mirnaName_{args.cl}"]
    mirNames = [mi.decode() for mi in mirNames]
    if chosen_m:
        chosen_m = list(set(chosen_m).intersection(set(mirNames)))
        indici = [mirNames.index(mi) for mi in chosen_m]
    else:
        indici = [True] * len(mirNames)

    y_train = trainfile[f"{args.cl}"][:]
    miRNA_train = trainfile[f"{args.cl}_mirna"][:]
    miRNA_train = miRNA_train[:, indici]
    geneName_train = trainfile["geneName"][:]
    geneName_train = [gene.decode() for gene in geneName_train]

    
    validfile = h5py.File(os.path.join(args.datadir, "valid.h5"), "r")
    y_valid = validfile[f"{args.cl}"][:]
    miRNA_valid = validfile[f"{args.cl}_mirna"][:]
    miRNA_valid = miRNA_valid[:, indici]
    geneName_valid = validfile["geneName"][:]
    geneName_valid = [gene.decode() for gene in geneName_valid]


    testfile = h5py.File(os.path.join(args.datadir, "test.h5"), "r")
    y_test = testfile[f"{args.cl}"][:]
    miRNA_test = testfile[f"{args.cl}_mirna"][:]
    miRNA_test = miRNA_test[:, indici]
    geneName_test = testfile["geneName"][:]
    geneName_test = [gene.decode() for gene in geneName_test]

    if args.verbose:
        print('Loaded target data')

    # Xpresso
    params = {
        "mir": args.mir,
        "datadir": args.datadir,
        "resultsdir": args.resultsdir,
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
        "dropout2": 0.01546
    }

    results = objective(params)
    print("Best Validation MSE = %.3f" % results["loss"])
    print(results)
    return results["tr_r2"]


def objective(params):
    global best_file
    global X_trainhalflife, y_train
    global miRNA_train, miRNA_valid
    global geneName_test, geneName_valid, geneName_train

    leftpos = int(params["leftpos"])
    rightpos = int(params["rightpos"])
    activationFxn = params["activationFxn"]

    X_trainpromoterSubseq = X_trainpromoter[:, leftpos:rightpos, :]
    X_validpromoterSubseq = X_validpromoter[:, leftpos:rightpos, :]

    halflifedata = Input(shape=(X_trainhalflife.shape[1:]), name="halflife")
    input_promoter = Input(shape=X_trainpromoterSubseq.shape[1:], name="promoter")

    x = Conv1D(
        int(params["numFiltersConv1"]),
        int(params["filterLenConv1"]),
        dilation_rate=int(params["dilRate1"]),
        padding="same",
        kernel_initializer="glorot_normal",
        input_shape=X_trainpromoterSubseq.shape[1:],
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

    miRNA = Input(shape=miRNA_train.shape[1:], name="mirna")
    if params["mir"]:
        x = Concatenate()([x, halflifedata, miRNA])
    else:
        x = Concatenate()([x, halflifedata])

    x = Dense(int(params["dense1"]))(x)
    x = Activation(activationFxn)(x)
    x = Dropout(params["dropout1"])(x)
    x = Dense(int(params["dense2"]))(x)
    x = Activation(activationFxn)(x)
    x = Dropout(params["dropout2"])(x)
    main_output = Dense(1)(x)

    if params["mir"]:
        model = Model(inputs=[input_promoter, halflifedata, miRNA], outputs=[main_output])
    else:
        model = Model(inputs=[input_promoter, halflifedata], outputs=[main_output])

    model.compile(SGD(lr=0.0005, momentum=0.9), "mean_squared_error", metrics=["mean_squared_error"])
    if args.verbose:
        print(model.summary())

    modelfile = os.path.join(params["resultsdir"], "plotted_model.png")
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=modelfile)

    check_cb = ModelCheckpoint(os.path.join(params["resultsdir"], "bestparams.h5"), 
    monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="min")
    if params["mir"]:
        result = model.fit([X_trainpromoterSubseq, X_trainhalflife, miRNA_train], y_train,
            batch_size=int(params["batchsize"]),
            shuffle="batch",
            epochs=100,
            validation_data=[[X_validpromoterSubseq, X_validhalflife, miRNA_valid], y_valid],
            callbacks=[check_cb, earlystop_cb])

    else:
        result = model.fit([X_trainpromoterSubseq, X_trainhalflife], y_train,
            batch_size=int(params["batchsize"]),
            shuffle="batch",
            epochs=100,
            validation_data=[[X_validpromoterSubseq, X_validhalflife], y_valid],
            callbacks=[check_cb, earlystop_cb])
    mse_history = result.history["val_mean_squared_error"]
    mse = min(mse_history)

    # evaluate performance on test set using best learned model
    best_file = os.path.join(params["resultsdir"], "bestparams.h5")
    model = load_model(best_file)

    with open(os.path.join(params["resultsdir"], "results.txt"), "w") as f:
        X_testpromoterSubseq = X_testpromoter[:, leftpos:rightpos, :]
        if params["mir"]:
            predictions_test = model.predict([X_testpromoterSubseq, X_testhalflife, miRNA_test], batch_size=64).flatten()
            predictions_train = model.predict([X_trainpromoterSubseq, X_trainhalflife, miRNA_train], batch_size=64).flatten()
            predictions_valid = model.predict([X_validpromoterSubseq, X_validhalflife, miRNA_valid], batch_size=64).flatten()
        else:
            predictions_test = model.predict([X_testpromoterSubseq, X_testhalflife], batch_size=64).flatten()
            predictions_train = model.predict([X_trainpromoterSubseq, X_trainhalflife], batch_size=64).flatten()
            predictions_valid = model.predict([X_validpromoterSubseq, X_validhalflife], batch_size=64).flatten()
        
        _, _, r_value_te, _, _ = stats.linregress(predictions_test, y_test)
        _, _, r_value_tr, _, _ = stats.linregress(predictions_train, y_train)
        _, _, r_value_val, _, _ = stats.linregress(predictions_valid, y_valid)
        
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
        df["Gene"] = np.concatenate((geneName_train, geneName_test, geneName_valid))
        df["Pred"] = np.concatenate((predictions_train, predictions_test, predictions_valid))
        df["Actual"] = np.concatenate((y_train, y_test, y_valid))
        df["Residual"] = df["Actual"] - df["Pred"]
        df.to_csv(os.path.join(params["resultsdir"], "predictions.txt"), index=False, header=True, sep="\t")

    return {"loss": mse, "tr_r2": r_value_tr ** 2}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='data/h5_datasets', type=str, help='Directory where the datasets are stored')
    parser.add_argument('--resultsdir', default='results', type=str, help='Directory where the results will be stored')
    parser.add_argument('--cl', default='mean', type=str, help='Category of tumour for training')
    parser.add_argument('--mir', action='store_true', help='Whether to use miRNA data or not')
    parser.add_argument('--mir_type', default='all', type=str, help='Which list of mirnas to use')
    parser.add_argument('--verbose', action='store_true', help='Print more output')
    parser.add_argument('--n_train', default=10, type=int, help='Number of different models to train')
    args = parser.parse_args()

    results_dir = (f"{args.resultsdir}/mirna/{args.mir_type}/{args.cl}" if args.mir else f"{args.resultsdir}/no_mirna/{args.cl}")
    args.resultsdir = results_dir
    if args.verbose:
        print('The run arguments are:')
        for arg in vars(args):
            print(arg, getattr(args, arg))
        print('\n')

    
    for i in range(args.n_train):
        tr = 0
        while tr < 0.01: # sometimes the model doesn't converge
            args.resultsdir = os.path.join(results_dir, f"{i}")
            if not os.path.exists(args.resultsdir):
                os.makedirs(args.resultsdir)
            if args.mir:
                if args.mir_type != "all":
                    with open(f"mirna_lists/{args.mir_type}_{args.cl}.txt") as file:
                        mirna_list = file.readlines()
                        mirna_list = [mirna.strip() for mirna in mirna_list]
                    print(f"Loading mirna_lists/{args.mir_type}_{args.cl}.txt")
                else:
                    mirna_list = False
            else:
                mirna_list = False
            tr = main(args, mirna_list)

