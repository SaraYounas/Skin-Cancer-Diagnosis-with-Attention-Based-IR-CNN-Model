# Skin-Cancer-Diagnosis-with-Attention-Based-IR-CNN-Model

## Description
This repository contains two Python notebooks that train two custom deep learning models AIR-CNN and IR-CNN. These models are designed for multiclass Skin Cancer Classification. The code includes data preprocessing, model training, and evaluation on test data.

## Dataset
Both models are trained on ISIC-19 Dataset

## How to Run
These notebooks are run on NVIDIA GeForce RTX 2080 Ti.
Code files Can be run Using Google Colab, Jupyter Notebook, VS Code or any other IDE. 
High Computational resources are required to run theses notebooks due to large datasset.


## Requirements
Before running the code, install and import following:

os, glob, numpy, pandas, matplotlib, cv2, tensorflow, Input,Model, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation,GlobalAveragePooling2D,Concatenate ,AveragePooling2D, BatchNormalization, LeakyReLU
, confusion_matrix, classification_report, train_test_split, keras, models, layers, regularizers, to_categorical, Dense, Input, Dropout, concatenate, Adam, plot_model, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Activation, Concatenate, Lambda, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, regularizers, activation, classification_report, confusion_matrix, roc_curve, auc


## Training Details
The notebooks include data loading, preprocessing, training, and evaluation steps.
Training parameters (e.g., batch size, learning rate, no of epochs) can be modified in both notebooks.


## Results
After training, the model checkpoints will be saved in the directory.

