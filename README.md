# nlpproject NLP Project Repository - Multistyle Text Style Transfer using GAN
## Students: S304046 SGRILLO MICHELE - S323173 LA ROTA FRANCESCO - S319845 BRACCO MATTEO
## Introduction

This GitHub repository contains files that are relevant for the Group 20 Project on Multistyle Text Style Transfer, using both the Amazon and Author datasets.

In particular, the repository is divided into 2 main folders:
1. Amazon
2. Author

Both folders contain:
1. Training data for the GAN: each style used for training, testing, and evaluation is in a separate file, for a total of 9 files for each dataset.
2. Colab Jupyter notebooks that were used to produce datasets, train classifiers, and train GANs.
3. Output CSV files.

*Note:* The Colab-specific paths have not been updated and are still set to the last version we used. For reproducibility purposes, they need to be updated.

## Amazon Dataset

It contains 3 subfolders described as follows:

### 00.Notebooks
This folder contains the code and is divided into 3 subfolders:
1. **00_Amazon_Preprocessing.jpynb** contains the code that was used to produce the datasets.
2. **01_TernaryClassifier_Training** contains the code that was used to train a ternary classifier with DeBERTA.
3. **02.GAN_Training** is a code example that can be used to replicate ternary training on each of the three style datasets.

### 01.Dataset
This folder contains the dataset files for both GAN training and ternary classifier training.  
Separate files for train, test, and evaluation for each style have been created, for a total of 9 files for the GAN.  
A comprehensive `train.csv`, `test.csv`, and `eval.csv` with the same samples is provided, which can be used to train and test the Ternary Classifier.

### 02.Output csv
It contains:
- **00.Neutral to positive and negative Experiment file**
- **01.Positive to neutral and negative Experiment file**

For both folders, the CSV files for G_AB and G_BA have been separated because G_AB moves from Style A to Styles B and C, whereas G_BA moves from Styles B and C to A.

## Author Dataset

It contains 3 subfolders described as follows. Note that the DeBERTA version of the code has not been attached, as it can be easily obtained from the provided code.

### 00.Dataset
This folder contains files that were created during the file preprocessing step; similar to the Amazon dataset, the sample GAN and classifier files have been separated.  
Separate files for train, test, and evaluation for each style have been created, for a total of 9 files for the GAN.  
A comprehensive `train.csv`, `test.csv`, and `eval.csv` with the same samples is provided, which can be used to train and test both Binary and Ternary classifiers.

### 01.Notebook
This folder contains both the summary notebook that was used for all the processing steps and the code folders that were called by the main notebook script.  
It also contains the following subfolders:
- **GAN_Binaria_shakespeare** contains the original publication code.
- **GAN_Tokenizer_Shakespeare** contains the enhanced tokenizer version of the GAN that enriches the original dictionary with context-specific words.

### 02.Output CSV
It contains training results and has been divided into three subfolders:
1. **00.Binary base** contains the original Binary GAN results applied to the new datasets; these files are our baseline.
2. **01.Binary Enhanced** contains Binary GAN results obtained by extending the generators' dictionaries with context-specific words.
3. **02.Ternary experiment Distilbert** contains ternary GAN results obtained by extending the dictionaries.
