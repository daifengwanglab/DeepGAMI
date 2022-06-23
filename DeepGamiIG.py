#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pramod
"""

import torch
import pandas as pd
import numpy as np
import scipy as sp
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from scipy import stats
import DeepGamiUtils as ut

device = 'cpu'

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

def get_mc_feat_importance(model, x1b, x2b, labels):
    """ Function to get feature importance for multi_class"""
    ig = IntegratedGradients(model)
    snp_imp, gene_imp = [], []
    for i in range(0, (max(labels)+1)):
        (snp_attr, gene_attr), approximation_error = ig.attribute((x1b, x2b),
                                                                  method='gausslegendre',
                                                                  return_convergence_delta=True,
                                                                  additional_forward_args='None',
                                                                  target=i)
        snp_attr = snp_attr.detach().numpy()
        gene_attr = gene_attr.detach().numpy()

        gene_imp.append(np.mean(abs(gene_attr), axis=0))
        snp_imp.append(np.mean(abs(snp_attr), axis=0))
    
    return snp_imp, gene_imp



 
""" prioritization analysis """

input_files = "demo/expMat_filtered.csv,demo/efeature_filtered.csv"
mid_phen_files = "None,None"
label_file = "demo/label_visual.csv"
model_file = "try/run_4_best_model.pth"
inp, _, labels = ut.get_mm_data(input_files, mid_phen_files, label_file, 'csv')

x1_np = np.log(inp[0]+1).to_numpy()[:, 0:1000]

inp[1] = inp[1].T
x2_np = preprocessing.scale(inp[1].to_numpy(),axis=0)

# defining model input tensors
snps_tr, gex_tr, y_tr = map(torch.tensor, (x1_np, x2_np, labels))
train_ds = TensorDataset(snps_tr, gex_tr, y_tr)
train_dl = DataLoader(dataset=train_ds, batch_size=3654, shuffle=True)
train_dl = WrappedDataLoader(train_dl, preprocess)

for x1b, x2b, yb in train_dl:
    break

print('Loading model...')
model = torch.load(model_file, map_location=torch.device('cpu'))
lbls = np.argmax(labels, 1)

gene_imp, ephys_imp = get_mc_feat_importance(model, x1b, x2b, lbls)

gene_list, ephys_list = list(inp[0].columns.values[0:1000]), list(inp[1].columns.values)
for i in range(len(gene_imp)):
    if i == 0:
        df1 = pd.DataFrame({'id':gene_list, 'L1': gene_imp[i]})
        df2 = pd.DataFrame({'id':ephys_list, 'L1': ephys_imp[i]})        
    else:
        df1['L'+str(i+1)] = gene_imp[i]
        df2['L'+str(i+1)] = ephys_imp[i]

print(df1.shape, df2.shape)
df1.to_csv('genes_prioritized.csv', index=False)
df2.to_csv('ephys_prioritized.csv', index=False)
