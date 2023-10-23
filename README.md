# miREx
The miREx GitHub repository provides the necessary code for predicting mRNA expression levels by incorporating miRNA expression data. The data used for this project is at https://zenodo.org/records/10033049.


## src directory
To set up the environment for executing the scripts, you can utilize the environment.yml file by running the following command:
```
   conda env create -f environment.yml
```
To download data from the genomic data commons portal, you need to install the gdc-client.
The binaries for different platforms are here:
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool.

The src directory includes these scripts:

1. gdc_download.py

   This script processes the manifests and JSONs to keep samples for which both mRNA and miRNA data are present. It downloads the data with the use of the gdc-client tool.

2. targetscan.py

   This script processes the file downloaded from targetscan, generating the ScoreMatrix.csv file. This file includes genes as rows, miRNAs as columns, and the corresponding efficacy scores of the miRNAs in repressing the respective mRNAs.

3. create_datasets.py
   
   This script takes the data downloaded from gdc and creates the .h5 files with normalized mRNA and miRNA expression values. If more subtypes are present, there will be an expression value for each subtype and their mean.

4. miREx_train.py

   This script manages the model training. The initial step involves training the baseline Xpresso model using the target mRNA data. For example, to train the model on the lung primary site (with the mean value of LUAD and LUSC subtypes), use the following command:
   ```
   python src/miREx_train.py --datadir path/to/datadir --resultsdir path/to/results/dir --primary_site lung --cancer_subtype mean
   ```
   The `--primary_site` parameter can be adjusted to train the model on tumors on different primary sites. 
   If different subtypes of that tumour exists, the `--cancer_subtype` parameter can be used to train on a particular subtype, otherwise the model will be trained using the mean of all subtypes. To train the model using all miRNAs, include the `--mir` argument:
   ```
   python src/miREx_train.py --datadir path/to/datadir --resultsdir path/to/results/dir --primary_site lung --cancer_subtype mean --mir
   ```

5. create_lists.py

   This script identifies the top 10 miRNAs with the highest correlation between residuals and targetscan CWCScore.
   The script miREx_train.py model can then be used with the `--mir_type` argument to choose which list of miRNAs to use for the training.
   ```
      python src/miREx_train.py --datadir path/to/datadir --resultsdir path/to/results/dir --primary_site lung --cancer_subtype mean --mir --mir_type corr
   ```
