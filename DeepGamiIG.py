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
    return inp1.float().to(device), inp2.float().to(device), oup.int().reshape(-1, 1).to(device)

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

def get_mc_feat_importance(model, x1b, x2b, x1_names, x2_names, labels):
    """ Function to get feature importance for multi_class"""
    ig = IntegratedGradients(model)
    x1_imp, x2_imp = [], []
    for i in range(0, (max(labels)+1)):
        (x1_attr, x2_attr), approximation_error = ig.attribute((x1b, x2b),
                                                                  method='gausslegendre',
                                                                  return_convergence_delta=True,
                                                                  additional_forward_args='None',
                                                                  target=i)
        x1_attr = x1_attr.detach().numpy()
        x2_attr = x2_attr.detach().numpy()

        x1_imp.append(np.mean(abs(x1_attr), axis=0))
        x2_imp.append(np.mean(abs(x2_attr), axis=0))
    
    for i in range(len(x1_attr)):
        if i == 0:
            df1 = pd.DataFrame({'id':x1_names, 'L1': x1_imp[i]})
            df2 = pd.DataFrame({'id':x2_names, 'L1': x2_imp[i]})        
        else:
            df1['L'+str(i+1)] = x1_imp[i]
            df2['L'+str(i+1)] = x2_imp[i]
    
    return x1_imp, x2_imp

def get_feat_importance(model, x1b, x2b, x1_names, x2_names, labels):
    """ Function to get feature importance """
    ig = IntegratedGradients(model)
    (x1_attr, x2_attr), approximation_error = ig.attribute((x1b, x2b),
                                                              method='gausslegendre',
                                                              return_convergence_delta=True,
                                                              additional_forward_args='None')
    x1_attr = x1_attr.detach().numpy()
    x2_attr = x2_attr.detach().numpy()

    x1_imp = pd.DataFrame({'id': x1_names, 'imp_score': np.mean(abs(x1_attr), axis=0)})
    x2_imp = pd.DataFrame({'id': x2_names, 'imp_score': np.mean(abs(x2_attr), axis=0)})

    return x1_imp, x2_imp


def get_layer_importance(model, x1b, x2b, target_names, labels):
    """ Function to get intemediate layer importance scores """
    cond = LayerConductance(model, model.drop_conn)
    inp_tg = cond.attribute((x1b, x2b), additional_forward_args='None')

    inp_tg = inp_tg.detach().numpy()
    tg_imp = pd.DataFrame({'target_id': target_names, 'imp_score': np.mean(abs(inp_tg), axis=0)})

    return tg_imp


def get_link_importance(model, x1b, x2b, labels, x1_names, x2_names, target_names):
    """ Function to extract the link importance between the input and the transparent layer"""
    
    neuron_cond = NeuronConductance(model, model.ffn)
    x1_all_neur_imp = pd.DataFrame()
    x2_all_neur_imp = pd.DataFrame()

    print('Interpreting eQTL and GRN connections importance...')
    for idx in list(range(0, len(target_names))):
        nc_vals = neuron_cond.attribute((x1b, x2b), additional_forward_args='None',
                                        neuron_selector=idx, attribute_to_neuron_input=True)

        x1_imp_neuron = pd.DataFrame({"id": x1_names,
                                      "imp_score": abs(nc_vals[0].mean(dim=0).detach().numpy())})
        x1_imp_neuron['target'] = target_names[idx]
        
        x2_imp_neuron = pd.DataFrame({"id": x2_names,
                                      "imp_score": abs(nc_vals[1].mean(dim=0).detach().numpy())})
        x2_imp_neuron['target'] = target_names[idx]

        
        if x1_all_neur_imp.shape[0] == 0:
            x1_all_neur_imp = x1_imp_neuron
            x2_all_neur_imp = x2_imp_neuron
        else:
            x1_all_neur_imp = pd.concat([x1_all_neur_imp, x1_imp_neuron])
            x2_all_neur_imp = pd.concat([x2_all_neur_imp, x2_imp_neuron])

    return x1_all_neur_imp, x2_all_neur_imp


""" Prioritization Analysis """
# Read the input data files
input_files = "demo/expMat_filtered.csv,demo/efeature_filtered.csv" # Point to the input files
mid_phen_files = "None,None" # Point to the intermediate files
label_file = "demo/label_visual.csv" # Point to the label files
inp, _, labels = ut.get_mm_data(input_files, mid_phen_files, label_file, 'csv')

# Standardization
x1_np = np.log(inp[0]+1).to_numpy()
x2_np = preprocessing.scale(inp[1].to_numpy(),axis=0)

# Get the feature names
x1_names, x2_names = list(inp[0].columns.values), list(inp[1].columns.values)

# Make data iterable
te_data1, te_data2, te_label = map(torch.tensor, (x1_np, x2_np, labels))
test_ds = TensorDataset(te_data1, te_data2, te_label)
test_dl = DataLoader(dataset=test_ds, batch_size=3654, shuffle=True)
test_dl = WrappedDataLoader(test_dl, preprocess)

for x1b, x2b, yb in test_dl:
    break

# Load the model
model_file = "try/run_4_best_model.pth" # Point to the trained model
print('Loading model...')
model = torch.load(model_file, map_location=torch.device('cpu'))

# Get prioritized features - IntegratedGradient
if len(set(labels)) > 2:
    feat1_imp, feat2_imp = get_mc_feat_importance(model, x1b, x2b, x1_names, x2_names, labels)
else:
    feat1_imp, feat2_imp = get_feat_importance(model, x1_names, x2_names, labels)
feat1_imp.to_csv('genes_prioritized.csv', index=False)
feat2_imp.to_csv('ephys_prioritized.csv', index=False)

# Get prioritized Intermeduate features - IntegratedGradient
if mid_phen_files.split(',')[0] != 'None':
    tg_file = 'data/processed/cmc_features/cmc_te_tgs.list' # Point to the target names file
    target_names = pd.read_csv(tg_file, header=None)
    target_names = list(target_names.iloc[:, 0])

    tg_imp = get_layer_importance(model, x1b, x2b, target_names, labels)
    tg_imp.to_csv('tgs_prioritized.csv', index=False)

# Get prioritized links
get_link_imp_flag = False
if get_link_imp_flag:
    feat1_neur_imp, feat2_neur_imp = get_link_importance(model, x1b, x2b, labels,
                                                         x1_names, x2_names, target_names)

    feat1_neur_imp.to_csv('snp_to_tg_links_prioritized.csv', index=False)
    feat2_neur_imp.to_csv('tf_to_tg_links_prioritized.csv', index=False)
