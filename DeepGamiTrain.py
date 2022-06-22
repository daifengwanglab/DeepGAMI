#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:17:02 2022
@author: pramod
"""

import argparse
import os
import time
import random
from collections import Counter
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sklearn.metrics as skm
from sklearn.model_selection import  StratifiedKFold, train_test_split, KFold
import DeepGamiUtils as ut
from DeepGamiModel import DeepGami

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
            yield self.func(*b)

def get_binary_performance(y_true, y_score):
    """Function to return the acc, bacc, and auc"""
    y_pred = np.argmax(y_score, 1)
    y_truth = np.argmax(y_true, 1)

    #auc = skm.roc_auc_score(y_true, y_score)
    acc = skm.accuracy_score(y_truth, y_pred)
    bacc = skm.balanced_accuracy_score(y_truth, y_pred)
    return acc, bacc#, auc

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
    """ Function to predict samples """
    predictions, truth = [], []
    for snps, gex, yb in data_dl:
        yhat = model(snps, gex, estimate)
        if task == 'binary':
            predictions.extend(yhat.detach().cpu().numpy())
            truth.extend(yb.detach().cpu().numpy())
        else:
            predictions.extend(torch.softmax(yhat, dim=1).detach().cpu().numpy())
            truth.extend(np.argmax(yb.detach().cpu().numpy(), 1))

    predictions = np.asarray(predictions)
    truth = np.asarray(truth)

    return predictions, truth

def train_step(model, loss_fn, data_dl, l1_reg, corr_reg, estimate, task, opt):
    """ Function to train an epoch"""
    tot_loss = 0.0
    predictions, truth = [], []
    corr_loss_fn = nn.MSELoss()

    for snps, gex, yb in data_dl:
        loss = 0.0

        yhat = model(snps, gex, estimate)
        _, _, Cg, Cg_est = model.get_intermediate_layers(snps, gex)

        # for param in modl.parameters():
        #     loss += l1_reg * torch.sum(torch.abs(param))

        pred_loss = loss_fn(yhat, yb.float())
        corr_loss = corr_loss_fn(Cg, Cg_est)
        loss += pred_loss + corr_reg*corr_loss

        if opt is not None:
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        if task == 'binary':
            predictions.extend(yhat.detach().cpu().numpy())
            truth.extend(yb.detach().cpu().numpy())
        else:
            predictions.extend(torch.softmax(yhat, dim=1).detach().cpu().numpy())
            truth.extend(np.argmax(yb.detach().cpu().numpy(), 1))
        tot_loss += loss.item()

    predictions = np.asarray(predictions)
    truth = np.asarray(truth)

    return tot_loss/len(data_dl), predictions, truth

def fit(epochs, model, loss_fn, opt, train_dl, val_dl, l1_reg, corr_reg, task, save_dir, cv_cntr):
    """ Function to fit the model """

    max_tr_acc, max_val_acc, max_val_cg_acc = 0, 0, 0
    max_tr_auc, max_val_auc, max_val_cg_auc = 0, 0, 0
    max_val_cs_acc, max_val_cs_auc = 0, 0

    max_tr_cl_acc, max_val_cl_acc, max_val_cg_cl_acc, max_val_cs_cl_acc = [], [], [], [] # Accuracy for each class
    stagnant, best_epoch = 0, 0

    # Iterate over several epochs. Ealry stopping criterira is applied
    for epoch in range(epochs):
        # Trainign phase - All modalities are given to the model
        model.train()
        estimate = 'None'
        tr_loss, tr_pred, tr_truth = train_step(model, loss_fn, train_dl, l1_reg,
                                                corr_reg, estimate, task, opt)

        # Evaluataion phase
        model.eval()
        # Both modality as input
        val_pred, val_truth = predict(model, val_dl, estimate, task)

        # Input is modality 1
        estimate = 'cg'
        val_cg_pred, _ = predict(model, val_dl, estimate, task)

        # Input is modality 2
        estimate = 'cs'
        val_cs_pred, _ = predict(model, val_dl, estimate, task)

        if task=='binary':
            tr_pred_bin = np.where(tr_pred<0.5, 0, 1)
            val_pred_bin = np.where(val_pred<0.5, 0, 1)
            val_cg_pred_bin = np.where(val_cg_pred<0.5, 0, 1)
            val_cs_pred_bin = np.where(val_cs_pred<0.5, 0, 1)
        else:
            tr_pred_bin = np.argmax(tr_pred, 1)
            val_pred_bin = np.argmax(val_pred, 1)
            val_cg_pred_bin = np.argmax(val_cg_pred, 1)
            val_cs_pred_bin = np.argmax(val_cs_pred, 1)


        tr_acc, tr_bacc, tr_auc = get_classification_performance(tr_truth, tr_pred, task)
        val_acc, val_bacc, val_auc = get_classification_performance(val_truth, val_pred, task)
        val_cg_acc, val_cg_bacc, val_cg_auc = get_classification_performance(val_truth,
                                                                             val_cg_pred, task)
        val_cs_acc, val_cs_bacc, val_cs_auc = get_classification_performance(val_truth,
                                                                             val_cs_pred, task)

        print("\n*** Epoch = %d ***"%(epoch))
        print("Training: Loss - %.4f, ACC - %.4f, BACC - %.4f, AUC - %.4f"%(tr_loss, tr_acc,
                                                                            tr_bacc, tr_auc))
        cfm = skm.confusion_matrix(tr_truth, tr_pred_bin)
        tr_cl_acc = cfm.diagonal()/cfm.sum(axis=1)
        print(tr_cl_acc)
        print(cfm)

        print("Validation: ACC - %.4f, BACC - %.4f, AUC - %.4f"%(val_acc, val_bacc, val_auc))
        cfm = skm.confusion_matrix(val_truth, val_pred_bin)
        val_cl_acc = cfm.diagonal()/cfm.sum(axis=1)
        print(val_cl_acc)
        print(cfm)


        print("Val Cs->Cg: ACC - %.4f, BACC - %.4f, AUC - %.4f"%(val_cg_acc, val_cg_bacc,
                                                                 val_cg_auc))
        cfm = skm.confusion_matrix(val_truth, val_cg_pred_bin)
        val_cg_cl_acc = cfm.diagonal()/cfm.sum(axis=1)
        print(val_cg_cl_acc)
        print(cfm)

        print("Val Cg->Cs: ACC - %.4f, BACC - %.4f, AUC - %.4f"%(val_cs_acc, val_cs_bacc,
                                                                  val_cs_auc))
        cfm = skm.confusion_matrix(val_truth, val_cs_pred_bin)
        val_cs_cl_acc = cfm.diagonal()/cfm.sum(axis=1)
        print(val_cs_cl_acc)
        print(cfm)

        if epoch == 0:
            max_tr_acc, max_val_acc = tr_bacc, val_bacc
            max_val_cg_acc, max_val_cs_acc = val_cg_bacc, val_cs_bacc
            max_tr_cl_acc, max_val_cl_acc  = tr_cl_acc, val_cl_acc
            max_val_cg_cl_acc, max_val_cs_cl_acc = val_cg_cl_acc, val_cs_cl_acc

            max_tr_auc, max_val_auc = tr_auc, val_auc
            max_val_cg_auc, max_val_cs_auc = val_cg_auc, val_cs_auc

            best_epoch = epoch
            torch.save(model, os.path.join(save_dir, 'run_' + str(cv_cntr) + '_best_model.pth'))

        else:
            #if (tr_bacc >= max_tr_acc) and (val_bacc > max_val_acc):
            if (val_cg_bacc > max_val_cg_acc) and (val_bacc > max_val_acc):
            #if (val_cg_bacc > max_val_cg_acc):
                max_tr_acc, max_val_acc = tr_bacc, val_bacc
                max_val_cg_acc, max_val_cs_acc = val_cg_bacc, val_cs_bacc
                max_tr_cl_acc, max_val_cl_acc  = tr_cl_acc, val_cl_acc
                max_val_cg_cl_acc, max_val_cs_cl_acc = val_cg_cl_acc, val_cs_cl_acc

                max_tr_auc, max_val_auc = tr_auc, val_auc
                max_val_cg_auc, max_val_cs_auc = val_cg_auc, val_cs_auc

                best_epoch = epoch
                torch.save(model, os.path.join(save_dir, 'run_' + str(cv_cntr) + '_best_model.pth'))

                print("saving model")
                stagnant = 0

            else:
                stagnant += 1
        if stagnant == 40:
            break

    perf_2m_dict = {'max_tr_acc': max_tr_acc, 'max_tr_auc': max_tr_auc,
                    'max_tr_cl_acc': max_tr_cl_acc, 'max_val_acc': max_val_acc,
                    'max_val_auc': max_val_auc, 'max_val_cl_acc': max_val_cl_acc}

    perf_1m_dict = {'max_val_cg_acc': max_val_cg_acc, 'max_val_cg_auc': max_val_cg_auc,
                    'max_val_cg_cl_acc': max_val_cg_cl_acc, 'max_val_cs_acc': max_val_cs_acc,
                    'max_val_cs_auc': max_val_cs_auc, 'max_val_cs_cl_acc': max_val_cs_cl_acc}

    return best_epoch, perf_2m_dict, perf_1m_dict

def run_cv_train(snp_data, gex_data, labels, args):
    """ Function to run cross validation modelling"""

    # Define the loss function and set the task
    if args.n_out == 1:
        args.task = 'binary'
        args.loss_fn = nn.BCELoss()
    else:
        args.task = 'multi_class'
        args.loss_fn = nn.CrossEntropyLoss()

    # Split dalta into 5 fold CV splits
    cv_k = 5
    rnd_seed = random.randint(1, 9999999)

    if args.task == 'binary':
        kfl = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=rnd_seed)
    else:
        kfl = KFold(n_splits=cv_k, shuffle=True, random_state=rnd_seed)

    cntr = 1
    tr_acc_sc, tr_auc_sc = [], []
    val_acc_sc, val_auc_sc = [], []
    val_cg_acc_sc, val_cg_auc_sc, = [], []
    val_cs_acc_sc, val_cs_auc_sc = [], []

    print("Random Seed = %d"%(rnd_seed))
    st_time = time.perf_counter()

    for tridx, teidx in kfl.split(labels if args.task=='binary' else np.argmax(labels, 1)):
        print("********** Run %d **********"%(cntr))

        snps_tr, snps_val = snp_data.values[tridx, :], snp_data.values[teidx, :]
        gex_tr, gex_val = gex_data.values[tridx, :], gex_data.values[teidx, :]

        if args.need_normalization:
            norm_method = list(args.norm_method.split(','))
            snps_tr = ut.normalize_data(snps_tr.copy(), args.norm_type, norm_method[0])
            snps_val = ut.normalize_data(snps_val.copy(), args.norm_type, norm_method[0])
            gex_tr = ut.normalize_data(gex_tr.copy(), args.norm_type, norm_method[1])
            gex_val = ut.normalize_data(gex_val.copy(), args.norm_type, norm_method[1])

        y_tr, y_val = labels[tridx] , labels[teidx]

        # Make data iterable with batches
        snps_tr, gex_tr, y_tr = map(torch.tensor, (snps_tr, gex_tr, y_tr))
        snps_val, gex_val, y_val = map(torch.tensor, (snps_val, gex_val, y_val))

        train_ds = TensorDataset(snps_tr, gex_tr, y_tr)
        val_ds = TensorDataset(snps_val, gex_val, y_val)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size*2, shuffle=False)

        train_dl = WrappedDataLoader(train_dl, preprocess)
        val_dl = WrappedDataLoader(val_dl, preprocess)

        # Get model parameters and create model
        fc_num_neurons = [int(x) for x in args.num_fc_neurons.split(',')]

        model = DeepGami(n_feat1=args.n_feat1, adj1=args.adj1, n_feat2=args.n_feat2,
                         adj2=args.adj2, latent_dim=args.latent_dim,
                         model_type=args.model_type, nfc=fc_num_neurons,
                         dkp=args.dropout_keep_prob, n_classes=args.n_out)

        model = model.to(device)
        if model is not None:
            for name, param in model.named_parameters():
                print(name, param.size())
            print(model)

        # Define the loss function snd initialize optimizer
        opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

        # Train the model
        _, perf_2m_dict, perf_1m_dict = fit(args.epochs, model, args.loss_fn, opt,
                                            train_dl, val_dl, args.out_reg,
                                            args.corr_reg, args.task, args.save, cv_cntr=cntr)

        tr_acc_sc.append(perf_2m_dict['max_tr_acc'])
        tr_auc_sc.append(perf_2m_dict['max_tr_auc'])

        val_acc_sc.append(perf_2m_dict['max_val_acc'])
        val_auc_sc.append(perf_2m_dict['max_val_auc'])

        val_cg_acc_sc.append(perf_1m_dict['max_val_cg_acc'])
        val_cg_auc_sc.append(perf_1m_dict['max_val_cg_auc'])

        val_cs_acc_sc.append(perf_1m_dict['max_val_cs_acc'])
        val_cs_auc_sc.append(perf_1m_dict['max_val_cs_auc'])

        tr_str = ",".join([str(x) for x in perf_2m_dict['max_tr_cl_acc']])
        val_str = ",".join([str(x) for x in perf_2m_dict['max_val_cl_acc']])
        val_cg_str = ",".join([str(x) for x in perf_1m_dict['max_val_cg_cl_acc']])
        val_cs_str = ",".join([str(x) for x in perf_1m_dict['max_val_cs_cl_acc']])

        with open(args.save + 'tr_2m_perf.csv', 'a') as f:
            f.write(tr_str + ',' + str(perf_2m_dict['max_tr_acc']) + '\n')

        with open(args.save + 'val_2m_perf.csv', 'a') as f:
            f.write(val_str + ',' + str(perf_2m_dict['max_val_acc']) + '\n')

        with open(args.save + 'val_1m_cg_perf.csv', 'a') as f:
            f.write(val_cg_str + ',' + str(perf_1m_dict['max_val_cg_acc']) + '\n')

        with open(args.save + 'val_1m_cs_perf.csv', 'a') as f:
            f.write(val_cs_str + ',' + str(perf_1m_dict['max_val_cs_acc']) + '\n')
        cntr += 1

    out_file = args.save + 'overall_perf.txt'
    header_str = "Model\tTrain BACC\tTrain AUC\tVal BACC\tVal AUC\tVal Cs->Cg BACC"
    header_str += "\tVal Cs->Cg AUC\tVal Cg->Cs BACC\tVal Cg->Cs AUC\n"

    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    wr_str = "%s\t%.5f +/- %.5f\t%.5f +/- %.5f" %(args.save, np.mean(tr_acc_sc), np.std(tr_acc_sc),
                                                  np.mean(tr_auc_sc), np.std(tr_auc_sc))
    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_acc_sc), np.std(val_acc_sc),
                                                 np.mean(val_auc_sc), np.std(val_auc_sc))
    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_cg_acc_sc), np.std(val_cg_acc_sc),
                                                 np.mean(val_cg_auc_sc), np.std(val_cg_auc_sc))
    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f\n" %(np.mean(val_cs_acc_sc), np.std(val_cs_acc_sc),
                                                   np.mean(val_cs_auc_sc), np.std(val_cs_auc_sc))

    with open(out_file, 'a') as write_fl:
        write_fl.write(wr_str)
        write_fl.close()

    # Keep the best model and remove the other folders
    fls = os.listdir(args.save)
    model_fls = [f for f in fls if f.startswith('run')]
    print(model_fls)
    keep_model = args.save + '/run_' + str(val_acc_sc.index(max(val_acc_sc))+1) + '_best_model.pth'
    for fls in model_fls:
        if (args.save + '/'+ fls) != keep_model:
            os.remove((args.save + '/' + fls))

    end_time = time.perf_counter()
    print("Five fold CV complete in %.3f minutes"%((end_time - st_time)/60.00))


def run_split_train(snp_data, gex_data, labels, args):
    """ Function to run train test split training"""

    # Define the loss function and set the task
    if args.n_out == 1:
        args.task = 'binary'
        args.loss_fn = nn.BCELoss()
    else:
        args.task = 'multi_class'
        args.loss_fn = nn.CrossEntropyLoss()

    tr_acc_sc, tr_auc_sc = [], []
    val_acc_sc, val_auc_sc = [], []
    val_cg_acc_sc, val_cg_auc_sc, = [], []
    val_cs_acc_sc, val_cs_auc_sc = [], []

    st_time = time.perf_counter()
    for i in range(0, args.n_iter):
        if args.need_normalization:
            norm_method = list(args.norm_method.split(','))
            x1_np = ut.normalize_data(snp_data.copy(), args.norm_type, norm_method[0])
            x2_np = ut.normalize_data(gex_data.copy(), args.norm_type, norm_method[1])
        else:
            x1_np = snp_data.copy().to_numpy()
            x2_np = gex_data.copy().to_numpy()

        X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(x1_np, x2_np, labels,
                                                                  test_size=0.20, random_state=i,
                                                                  stratify=np.argmax(labels, 1))

        if args.oversampling:
            data_tr, y_tr = ut.perform_oversampling(np.concatenate((X1_tr, X2_tr), axis=1),
                                                     np.argmax(y_tr, 1), random_state = i)
            X1_tr = data_tr[:, 0:args.n_feat1]
            X2_tr = data_tr[:, args.n_feat1:(args.n_feat1+args.n_feat2)]
        print(Counter(np.argmax(y_tr, 1)))

        # Make data iterable with batches
        X1_tr, X2_tr, y_tr = map(torch.tensor, (X1_tr, X2_tr, y_tr))
        X1_te, X2_te, y_te = map(torch.tensor, (X1_te, X2_te, y_te))

        train_ds = TensorDataset(X1_tr, X2_tr, y_tr)
        val_ds = TensorDataset(X1_te, X2_te, y_te)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size*2, shuffle=False)

        train_dl = WrappedDataLoader(train_dl, preprocess)
        val_dl = WrappedDataLoader(val_dl, preprocess)

        # Get model parameters and create model
        fc_num_neurons = [int(x) for x in args.num_fc_neurons.split(',')]

        model = DeepGami(n_feat1=args.n_feat1, adj1=args.adj1, n_feat2=args.n_feat2,
                         adj2=args.adj2, latent_dim=args.latent_dim,
                         model_type=args.model_type, nfc=fc_num_neurons,
                         dkp=args.dropout_keep_prob, n_classes=args.n_out)

        model = model.to(device)
        if model is not None:
            for name, param in model.named_parameters():
                print(name, param.size())
            print(model)

        # Define the loss function snd initialize optimizer
        opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

        # Train the model
        _, perf_2m_dict, perf_1m_dict = fit(args.epochs, model, args.loss_fn, opt,
                                            train_dl, val_dl, args.out_reg,
                                            args.corr_reg, args.task, args.save, cv_cntr=(i+1))
        end_time = time.perf_counter()
        print("Iter %d complete in %.3f minutes"%(i, (end_time - st_time)/60.00))

        tr_acc_sc.append(perf_2m_dict['max_tr_acc'])
        tr_auc_sc.append(perf_2m_dict['max_tr_auc'])

        val_acc_sc.append(perf_2m_dict['max_val_acc'])
        val_auc_sc.append(perf_2m_dict['max_val_auc'])

        val_cg_acc_sc.append(perf_1m_dict['max_val_cg_acc'])
        val_cg_auc_sc.append(perf_1m_dict['max_val_cg_auc'])

        val_cs_acc_sc.append(perf_1m_dict['max_val_cs_acc'])
        val_cs_auc_sc.append(perf_1m_dict['max_val_cs_auc'])

        tr_str = ",".join([str(x) for x in perf_2m_dict['max_tr_cl_acc']])
        val_str = ",".join([str(x) for x in perf_2m_dict['max_val_cl_acc']])
        val_cg_str = ",".join([str(x) for x in perf_1m_dict['max_val_cg_cl_acc']])
        val_cs_str = ",".join([str(x) for x in perf_1m_dict['max_val_cs_cl_acc']])

        with open(args.save + 'tr_2m_perf.csv', 'a') as f:
            f.write(tr_str + ',' + str(perf_2m_dict['max_tr_acc']) + '\n')

        with open(args.save + 'val_2m_perf.csv', 'a') as f:
            f.write(val_str + ',' + str(perf_2m_dict['max_val_acc']) + '\n')

        with open(args.save + 'val_1m_cg_perf.csv', 'a') as f:
            f.write(val_cg_str + ',' + str(perf_1m_dict['max_val_cg_acc']) + '\n')

        with open(args.save + 'val_1m_cs_perf.csv', 'a') as f:
            f.write(val_cs_str + ',' + str(perf_1m_dict['max_val_cs_acc']) + '\n')

    out_file = args.save + 'overall_perf.txt'
    header_str = "Model\tTrain BACC\tTrain AUC\tVal BACC\tVal AUC\tVal Cs->Cg BACC"
    header_str += "\tVal Cs->Cg AUC\tVal Cg->Cs BACC\tVal Cg->Cs AUC\n"

    if not os.path.exists(out_file):
        with open(out_file, 'w') as write_fl:
            write_fl.write(header_str)
            write_fl.close()

    wr_str = "%s\t%.5f +/- %.5f\t%.5f +/- %.5f" %(args.save, np.mean(tr_acc_sc), np.std(tr_acc_sc),
                                                  np.mean(tr_auc_sc), np.std(tr_auc_sc))
    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_acc_sc), np.std(val_acc_sc),
                                                 np.mean(val_auc_sc), np.std(val_auc_sc))
    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f" %(np.mean(val_cg_acc_sc), np.std(val_cg_acc_sc),
                                                 np.mean(val_cg_auc_sc), np.std(val_cg_auc_sc))
    wr_str += "\t%.5f +/- %.5f\t%.5f +/- %.5f\n" %(np.mean(val_cs_acc_sc), np.std(val_cs_acc_sc),
                                                   np.mean(val_cs_auc_sc), np.std(val_cs_auc_sc))

    with open(out_file, 'a') as write_fl:
        write_fl.write(wr_str)
        write_fl.close()

    fls = os.listdir(args.save)
    model_fls = [f for f in fls if f.startswith('run')]
    print(model_fls)
    print(val_acc_sc)
    keep_model = args.save + '/run_' + str(val_acc_sc.index(max(val_acc_sc))+1) + '_best_model.pth'
    for fls in model_fls:
        if (args.save + '/'+ fls) != keep_model:
            os.remove((args.save + '/' + fls))

    end_time = time.perf_counter()
    print("%d iterations complete in %.3f minutes"%(args.n_iter, (end_time - st_time)/60.00))

def train_deepgami(args):
    """ Method to fetch the data and perfrom training """
    print("hello")

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    #-- Load and preprocess data -- #
    # Fetch required inputs
    st_time = time.perf_counter()
    inp, adj, labels = ut.get_mm_data(args.input_files, args.intermediate_phenotype_files,
                                      args.disease_label_file, 'csv')

    if inp[0].shape[0] != inp[1].shape[0]:
        inp[1] = inp[1].T

    args.n_out = labels.shape[1]
    print('Input 1 shape', inp[0].shape)
    print('Input 2 shape', inp[1].shape)

    args.n_out = labels.shape[1]

    if args.model_type == 'drop_connect':
        args.n_feat1, args.latent_dim = adj[0].shape
        args.n_feat2 = adj[1].shape[0]
        args.adj1 = torch.from_numpy(adj[0]).float().to(device)
        args.adj2 = torch.from_numpy(adj[1]).float().to(device)
    else:
        args.n_feat1, args.n_feat2 = inp[0].shape[1], inp[1].shape[1]
        args.adj1, args.adj2 = None, None

    end_time = time.perf_counter()
    print("Data fetch & split completed in %.3f mins\n"%((end_time - st_time)/60.00))

    # Training
    if args.cross_validate:
        run_cv_train(inp[0], inp[1], labels, args)
    else:
        run_split_train(inp[0], inp[1], labels, args)


def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--num_data_modal', type=int, default=2,
                        help='Path to the input data file')
    parser.add_argument('--input_files', type=str,
                        default="data/rosmap/cogdx/rosmap_geno.csv,data/rosmap/cogdx/rosmap_gex.csv",
                        help='Comma separated input data paths')
    parser.add_argument('--intermediate_phenotype_files', type=str,
                        default="None,None",
                        help='Path to transparent layer adjacency matrix')
    parser.add_argument('--disease_label_file', type=str,
                        default="data/rosmap/cogdx/rosmap_cogdx_labels.csv",
                        help='Path to Output labels file - Disease phenotypes')

    # Hyper parameters
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--out_reg', type=float, default=0.005, help='l2_reg_lambda')
    parser.add_argument('--corr_reg', type=float, default=0.5, help='l2_corr_lambda')

    # First transparent layer
    parser.add_argument('--model_type', type=str, default='fully_connect',
                        help='Drop Connect vs FCN vs both for the first transparent layer')
    parser.add_argument('--latent_dim', type=int, default=500,
                        help='Number of dimensions for the latent space to be reduced.')

    # FCN
    parser.add_argument('--num_fc_neurons', type=str, default='350,200',
                        help='Number of kernels for fully connected layers, comma delimited.')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Droupout % for handling overfitting. 1 to keep all & 0 to keep none')

    # Settings
    # Data normalization
    parser.add_argument('--need_normalization', type=bool, default=True,
                        help='Flag to perfrom data normalization')
    parser.add_argument('--norm_method', type=str, default='standard,standard',
                        help='Standard vs log vs min-max for each input dataset. comma-separated')
    parser.add_argument('--norm_type', type=str, default='features',
                        help='Feature normalization vs sample normalization')

    # Data split
    parser.add_argument('--train_percent', type=float, default=0.8,
                        help='Choose how the tain and testvalidation split to occur.')
    parser.add_argument('--need_balance', type=bool, default=False, help='balanced_training')

    # Model training
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--stagnant', type=int, default=100, help='Early stop criteria')
    parser.add_argument('--n_iter', type=int, default=3, help='n_iter')
    parser.add_argument('--oversampling', type=bool, default=False,
                        help='Flag to perfrom oversampling based on the unbalanced data')
    parser.add_argument('--cross_validate', type=bool, default=True,
                        help='Choose normal validation vs cross-validation')

    # Model save paths
    parser.add_argument('--save', type=str, default="model/ROSMAP_cogdx_fc_try_cv/",
                        help="path to save model")

    # Remove these parameters later. This is for our convenience
    parser.add_argument('--split_sample_ids', type=str, help="training and testing splits",
                        default="None")


    args = parser.parse_args()
    print(args)
    train_deepgami(args)

if __name__ == '__main__':
    main()
