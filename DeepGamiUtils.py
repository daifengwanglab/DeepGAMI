#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:38:49 2022
@author: pramod
"""

import sys
from collections import Counter
import pandas as pd
import numpy as np
import scipy as sp
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

def perform_oversampling(data, labels, random_state):
    """ mehod to perfrom oversampling of training data """
    data_new, y_new = SMOTE(random_state=random_state).fit_resample(data, labels)
    unique, inverse = np.unique(y_new, return_inverse=True)
    return data_new, np.eye(unique.shape[0])[inverse]

def normalize_data(data, norm_type='features', norm_method='standard'):
    """ Function to normalize data """
    if norm_method == 'log':
        return np.log(data+1)
    else:
        scaler = None
        if norm_method == 'minmax':
            scaler = preprocessing.MinMaxScaler()
        elif norm_method == 'standard':
            scaler = preprocessing.StandardScaler()

        if norm_type == 'features':
            data = scaler.fit_transform(data)
        else:
            data_t = scaler.fit_transform(data.T)
            data = data_t.T
        return data

def file_check(args):
    """ Function to check if all required file locations are provided by the user"""
    input_files = list(args.input_files.split(','))
    adj_files = list(args.intermediate_phenotype_files.split(','))

    if any([args.num_data_modal != len(input_files), args.num_data_modal != len(adj_files)]):
        print("Error number of data modes and the corresponding files do not match")
        sys.exit(1)
    return True

def one_hot_encoding(labels):
    """ One hot encoding of the labels """

    if '02/03/22' in labels:
        labels[labels == '02/03/22'] = '2/3'
    unique, inverse = np.unique(labels, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

def get_mm_data(input_files, mid_phenotype_files, label_file, file_format):
    """ Read and fetch multi-modal data from files"""

    inp_files = list(input_files.split(','))
    adj_files = list(mid_phenotype_files.split(','))

    inp, adj = [], []

    for i in range(len(inp_files)):
        if file_format =='csv':
            dm_inp = pd.read_csv(inp_files[i], header=0)#.drop(columns=['Unnamed: 0'])
            lbls = pd.read_csv(label_file, header=0)#.drop(columns=['Unnamed: 0'])
        else:
            dm_inp = pd.read_pickle(inp_files[i])
            lbls = pd.read_pickle(label_file)

        dm_inp = dm_inp.set_index(dm_inp.columns[0])
        inp.append(dm_inp.T)
        print("dm_inp_%d"%i, dm_inp.shape)

        if mid_phenotype_files != 'None':
            adj_sp = sp.sparse.load_npz(adj_files[i])
            dm_adj = adj_sp.todense()
            #dm_adj[dm_adj == 0] = np.max(dm_adj)/10.0
            adj.append(dm_adj)
            print("dm_adj_%d"%i, dm_adj.shape)

    labels = lbls['label'].values
    print(Counter(labels))
    if max(labels) > 1:
        labels = one_hot_encoding(labels)
    else:
        labels = labels.reshape(-1, 1)
    print("labels", labels.shape)

    return inp, adj, labels
