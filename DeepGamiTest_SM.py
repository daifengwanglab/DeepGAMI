#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pramod
"""

import sys
import argparse
import torch
from torch import nn
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
from DeepGamiModel import DeepGami
import warnings
warnings.simplefilter("ignore")

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
    
    return predictions

def run_test(args):
    """ Function to test single modality outcomes """
    model = torch.load(args.model_file, map_location=torch.device(device))

    # modality 1
    snp_data = pd.read_csv(args.input_file)
    snp_data = snp_data.set_index(snp_data.columns[0])

    if model.fcn1.in_features != snp_data.shape[1]:
        snp_data = snp_data.T
        
        if model.fcn1.in_features != snp_data.shape[1]:
            sys.exit("Feature mismatch.")

    # modality 2 ==> change to ones
    gex_data = torch.ones(snp_data.shape[0], model.fcn2.in_features)

    scaler = preprocessing.StandardScaler()
    snps_te = scaler.fit_transform(snp_data)
    gex_te = scaler.fit_transform(gex_data)

    snps_te, gex_te = map(torch.tensor, (snps_te, gex_te))
    te_ds = TensorDataset(snps_te, gex_te)
    te_dl = DataLoader(dataset=te_ds, batch_size=100, shuffle=False)

    te_dl = WrappedDataLoader(te_dl, preprocess)

    # cg -> Estimates Cg from Cs
    pred = predict(model, te_dl, 'cg', args.task)

    # Write a file with the patient IDs and the probabilities each has Alzheimer's
    pred_cols = ['Class'+str(i) + ' Score' for i in range(1, (pred.shape[1]+1))]
    pred_df = pd.DataFrame.from_records(pred)
    pred_df.columns = pred_cols
    pred_df.insert(0, 'id', list(snp_data.index))
    pred_df.to_csv("test_class_scores.csv")
    

    print("")
    print("Printing class label scores for 10 samples. Scores for all samples is saved in test_class_scores.csv file\n")
    print(pred_df.head(10))
    
    if args.label_file is not None:
        # Read phenotype file
        lbls = pd.read_csv(args.label_file)
        labels = lbls.label.values
        acc, bacc, auc = get_classification_performance(labels, pred, args.task)

        print("The prediction performance for the given test samples is as follows:")
        print("BACC = ", bacc)
        print("AUC = ", auc)

def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the input data modality you wish to test.')
    parser.add_argument('--label_file', type=str, default=None,
                        help='The 1s and 0s dictating positive and negative cases, respectively.')
    parser.add_argument('--model_file', type=str, default='run_92_best_model.pth',
                        help='The trained model.')
    parser.add_argument('--task', type=str, default='binary',
                        help='Choose between binary and multiclass')

    args = parser.parse_args()
    run_test(args)

if __name__ == '__main__':
    main()
