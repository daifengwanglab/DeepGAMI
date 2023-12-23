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

def preprocess(inp1, inp2):
    """ Function to direct the input and ouput to CPU vs GPU"""
    return inp1.float().to(device), inp2.float().to(device)

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
    for snps, gex in data_dl:
        yhat = model(snps, gex, estimate)
        if task == 'binary':
            predictions.extend(yhat.detach().cpu().numpy())
        else:
            predictions.extend(torch.softmax(yhat, dim=1).detach().cpu().numpy())
    predictions = np.asarray(predictions)
    return predictions, truth

def run_test(args):
    """ Function to test single modality outcomes """
    model = torch.load(args.model_file, map_location=torch.device(device))
    
    # modality 1
    snp_data = pd.read_csv(args.input_file).set_index('SubID')

    # modality 2
    gex_data = torch.ones(snp_data.shape[0], model.fcn2.in_features)

    scaler = preprocessing.StandardScaler()
    snps_te = scaler.fit_transform(snp_data)
    gex_te = scaler.fit_transform(gex_data)

    snps_te, gex_te = map(torch.tensor, (snps_te, gex_te))
    te_ds = TensorDataset(snps_te, gex_te)
    te_dl = DataLoader(dataset=te_ds, batch_size=100, shuffle=False)

    te_dl = WrappedDataLoader(te_dl, preprocess)

    # cg -> Estimates Cg from Cs
    pred = predict(model, te_dl, 'cg', 'binary')

    # Write a file with the patient IDs and the probabilities each has Alzheimer's
    patient_preds = []
    for p in pred:
        patient_preds.append("{:.5f}".format(p[0]))

    patient_dict = {'sample_id': snp_data.index, 'AD Score': patient_preds} 
    patient_df = pd.DataFrame(patient_dict)

    patient_df.to_csv("sample_AD_scores.csv")

    print("Printing AD scores for top 10 samples. Scores for all samples is saved in sample_AD_scores.csv file\n")
    print(patient_df.head(10))

    if args.labels is not None:
        # Read phenotype file
        lbls = pd.read_csv(args.label_file)
        labels = lbls.label.values
        acc, bacc, auc = get_classification_performance(labels, pred)

        print("The prediction performance for the given test samples is as follows:")
        print("BACC =", bacc)
        print("AUC =", auc)

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the single input modailty data file with available')
    parser.add_argument('--label_file', type=str, default=None,
                        help='Path to the label file. Must contain a column named label')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Path to the trained model location.')

    args = parser.parse_args()
    run_test(args)

if __name__ == '__main__':
    main()
