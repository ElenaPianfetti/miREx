# miREx

The miREx GitHub repository provides the necessary code and data for predicting mRNA expression levels by incorporating miRNA expression data.

## data directory
In the data directory, you can find the datasets containing mRNA and miRNA expression levels for two tumor classes, LUAD and LUSC, as well as the average expression levels across the two classes. The directory also includes the data downloaded from targetscan.

The data about the promoter sequence and the halflife features has to be downloaded from the Xpresso repository.

## src directory
To set up the environment for executing all the scripts, you can utilize the environment.yml file by executing the following command:
```
   conda env create -f environment.yml
```
The src directory includes four scripts:
1. targetscan.py: This script processes the file downloaded from targetscan, generating the ScoreMatrix.csv file. This file includes genes as rows, miRNAs as columns, and the corresponding efficacy scores of the miRNAs in repressing the respective mRNAs.
2. miREx_train.py: This script manages the model training. The initial step involves training the baseline Xpresso model using the target mRNA data. To train the model on the average of LUAD and LUSC classes, use the following command:
   ```
   python src/miREx_train.py --datadir path/to/datadir --resultsdir path/to/results/dir --cl mean
   ```
   The `--cl` parameter can be adjusted to train the model on the other two classes. To train the model using all miRNAs, include the `--mir` argument:
   ```
   python src/miREx_train.py --datadir path/to/datadir --resultsdir path/to/results/dir --cl mean --mir
   ```

3. create_lists.py: This script identifies the top 10 miRNAs with the highest correlation between residuals and targetscan CWCScore. It requires the path of the best models obtained during training as input.
The script miREx_train.py model can then be used with the --mir_type argument, to choose which list of miRNAs to use for the training.
```
   python src/miREx_train.py --datadir path/to/datadir --resultsdir path/to/results/dir --cl mean --mir --mir_type corr
```

4. plot.py: Once the baseline models, models with all miRNAs, and models with the highest correlated miRNAs are trained, this script visualizes the results. It also generates a results.csv file, which provides an overview of all the results from different runs.

## mirna_lists directory
The mirna_lists directory contains six lists created using the create_lists.py script. There are three lists for correlation to the residuals (corr_mean.txt, corr_LUAD.txt, and corr_LUSC.txt) and three lists for correlation to the absolute value of the residuals (corr_abs_mean.txt, corr_abs_LUAD.txt, and corr_abs_LUSC.txt).