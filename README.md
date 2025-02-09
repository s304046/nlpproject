# nlpproject NLP Project Repository - Multistyle Text Style Transfer using GAN
## Student: S304046 SGRILLO MICHELE - S323173 LA ROTA FRANCESCO - S319845 BRACCO MATTEO
## Introduction 

This Git hub contains files that are more relevant for the Group 20 Project

In particular the GIT has been divided into 2 main folders:
1. Amazon
2. Author

Both folders contains:
1. Train data for the GAN: each style considered for train, test and eval is in a separate file for a total of 9 files for each Work
2. Colab jupyther notebooks that have been used to produce datasets, train the classifier and train the GAN.
3. Output CSV

## Amazon Dataset 
00.Notebooks for Amazon Folder contains:
1. 00_Amazon_Preprocessing.jpynb contains the code that has been used to produce datasets
2. 01_TernaryClassifier_Training contains the code that has been used to train ternary classifier with DeBERTA
3. 02.GAN_Training code example that be used to replicate a ternary training on each Three Style Dataset

01.Dataset contains:
Separate file for train, test and eval for each style for a total of 9 files
A comprehensive train.csv, test.csv, eval.csv with same samples  that can be used to train and test the Ternary Classifier

02.Output csv contains 
00.Neutral to positive and negative Experiment file
01.Positive to neutral and negative Experiment file

For both folders G_AB and G_BA .csv have been separated because G_ab moves from Styles A to B and C 
Meanwhile G_BA moves from B and C to A

In both cases style style tokens with required directions should be prepended to the file. 

## Author Dataset 
00.Dataset contains files that have created during file preprocessing step: like the Amazon sample GAN and Classifier Files have been separated

01.Notebook contains a single notebook with all steps that have been done starting from original data preprocessing to ternary GAN training

02.Output CSV has been divided into three subfolders:
1. 00.Binary base contains the original Binary GAN results applied on the new datasets: these files are out baseline for the experiment
2. 01.Binary Enhanced contains Binary GAN results obtained extending the Dictionaries of Generators with context specific words
3. 02.Ternary experiment Distilbert contains ternary GAN results extending the Dictionaries of 


