# nlpproject NLP Project Repository - Multistyle Text Style Transfer using GAN
## Student: S304046 SGRILLO MICHELE - S323173 LA ROTA FRANCESCO - S319845 BRACCO MATTEO
## Introduction 

This Git hub contains files that are more relevant for the Group 20 Project of Multistyle Text Style Transfer using both Amazon and Author Dataset

In particular the GIT has been divided into 2 main folders:
1. Amazon
2. Author

Both folders contains:
1. Train data for the GAN: each style considered for train, test and eval is in a separate file for a total of 9 files for each Work
2. Colab jupyther notebooks that have been used to produce datasets, to train  classifiers and to train GANs.
3. Output CSV

As a general indication "colab" specific paths have not been updated and are still in the last version we used them.
For reproducibility purposes they need to be updated. 

## Amazon Dataset

It cointains 3 subfolders described as it follows: 

### 00.Notebooks
It contains the code and it is divided into 3 subfolders:
1. 00_Amazon_Preprocessing.jpynb contains the code that has been used to produce datasets
2. 01_TernaryClassifier_Training contains the code that has been used to train ternary classifier with DeBERTA
3. 02.GAN_Training code example that be used to replicate a ternary training on each Three Style Dataset

### 01.Dataset
It contains the dataset files both for the GAN Training and ternary classifier training. 
Separate files for train, test, eval and  for each style for a total of 9 files have been created for the GAN. 
A comprehensive train.csv, test.csv, eval.csv with same samples  that can be used to train and test the Ternary Classifier

### 02.Output csv
It contains 
00.Neutral to positive and negative Experiment file
01.Positive to neutral and negative Experiment file
For both folders G_AB and G_BA .csv have been separated because G_ab moves from Styles A to B and C; meanwhile G_BA moves from B and C to A

## Author Dataset 

It cointains 3 subfolders described as it follows. As a general indication the Deberta version of the code has not been attached since it can be easily be obtained from the attached one.  

### 00.Dataset
It contains files that have created during file preprocessing step: like the Amazon sample GAN and Classifier Files have been separated.
Separate files for train, test, eval and  for each style for a total of 9 files have been created for the GAN. 
A comprehensive train.csv, test.csv, eval.csv with same samples that can be used to train and test both Binary and Ternary classifiers.

### 01.Notebook
It contains both the summary notebook that has been used for all steps that have been done and code folders that have been called by main notebook script. 
It  also contains the attached subfolders:
GAN_Binaria_shakespeare
GAN_Tokenizer_Shakespeare

GAN_Binaria_shakespeare contains the original publication code
GAN_Tokenizer_Shakespeare contains the enhanced tokenizer version of GAN that enriches the original dictionary with context specific words. 

### 02.Output CSV 
It contains train results and has been divided into three subfolders:
1. 00.Binary base contains the original Binary GAN results applied on the new datasets: these files are our baseline;
2. 01.Binary Enhanced contains Binary GAN results obtained extending the Dictionaries of Generators with context specific words;
3. 02.Ternary experiment Distilbert contains ternary GAN results extending the Dictionaries; 


