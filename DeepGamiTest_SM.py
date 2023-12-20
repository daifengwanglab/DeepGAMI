#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 2022
@author: pramod
"""

import sys
import argparse
import os
import time
import random
import torch
from torch import nn
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.metrics as skm
from sklearn.model_selection import  StratifiedKFold, train_test_split, KFold
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
import DeepGamiUtils as ut
from DeepGamiModel import DeepGami
#from DeepDiceMVModel import DeepDiceMV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score 
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(inp1, inp2, oup):
    """ Function to direct the input and ouput to CPU vs GPU"""
    return inp1.float().to(device), inp2.float().to(device), oup.int().to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def get_classification_performance(y_true, y_score, task='binary'):
    """Function to return various performance metrics"""
    auc = 0.0

    y_pred = None
    if task=='binary':
        auc = skm.roc_auc_score(y_true, y_score)
        y_pred = np.where(y_score<0.5, 0, 1)
    else:
        auc = skm.roc_auc_score(y_true, y_score, average="weighted", multi_class="ovr")
        y_pred = np.argmax(y_score, 1)

    acc = skm.accuracy_score(y_true, y_pred)
    bacc = skm.balanced_accuracy_score(y_true, y_pred)
    return acc, bacc, auc

def predict(model, data_dl, estimate, task):
    """ Function to predict the samples """
    predictions, truth = [], []
    for snps, gex, yb in data_dl:
        yhat = model(snps, gex, estimate)

        if task == 'binary':
            predictions.extend(yhat.detach().cpu().numpy())
            truth.extend(yb.detach().cpu().numpy())
        else:
            predictions.extend(torch.softmax(yhat, dim=1).detach().cpu().numpy())
            truth.extend(yb.detach().cpu().numpy())
    
    predictions = np.asarray(predictions)
    truth = np.asarray(truth)
    return predictions, truth

def run_test(args):
    """ Function to test single modality outcomes """
    modality = args.input_modality
    phenotype = args.labels
    trained_model = args.trained_model

    model_file = args.trained_model
    model = torch.load(model_file, map_location=torch.device(device))
    
    # modality 1
    snp_file = './' + modality
    snp_data = pd.read_csv(snp_file).set_index('SubID')

    # modality 2 ==> change to ones
    gex_data = torch.ones(snp_data.shape[0], model.fcn2.in_features)

    lbl_file = phenotype
    lbls = pd.read_csv(lbl_file)
    labels = lbls.label.values

    scaler = preprocessing.StandardScaler()
    snps_te = scaler.fit_transform(snp_data)
    gex_te = scaler.fit_transform(gex_data)

    snps_te, gex_te, y_te = map(torch.tensor, (snps_te, gex_te, labels))
    te_ds = TensorDataset(snps_te, gex_te, y_te)
    te_dl = DataLoader(dataset=te_ds, batch_size=100, shuffle=False)

    te_dl = WrappedDataLoader(te_dl, preprocess)
    pred, truth = predict(model, te_dl, 'cg', 'binary')

    acc, bacc, auc = get_classification_performance(truth, pred)

    print("BACC =", bacc)
    print("AUC =", auc)


def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--input_modality', type=str, default=None,
                        help='Path to the input data modality you wish to test.')
    parser.add_argument('--labels', type=str, default=None,
                        help='The 1s and 0s dictating positive and negative cases, respectively.')
    parser.add_argument('--trained_model', type=str, default=None,
                        help='The trained model.')

    args = parser.parse_args()
    run_test(args)

if __name__ == '__main__':
    main()