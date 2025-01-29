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
Before running the code, install the necessary librarires

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation,GlobalAveragePooling2D,Concatenate ,AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import models, layers, regularizers
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Input, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras import backend
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import regularizers, activations
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


## Training Details
The notebooks include data loading, preprocessing, training, and evaluation steps.
Training parameters (e.g., batch size, learning rate, no of epochs) can be modified in both notebooks.


## Results
After training, the model checkpoints will be saved in the directory.

