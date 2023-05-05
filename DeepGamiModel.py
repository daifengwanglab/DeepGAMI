#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pbchandr
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

class FNN(nn.Module):
    """ Class for Feed Forward Network"""
    def __init__(self, input_dim, hidden_dim_array, dropout_keep_prob):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_layers = len(hidden_dim_array)
        for idx in range(self.hidden_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim_array[idx]))
            if self.hidden_layers >= 1 and idx < (self.hidden_layers - 1):
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(1-dropout_keep_prob))
            input_dim = hidden_dim_array[idx]

    def forward(self, inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp

class DropConnect(nn.Module):
    """ Class for performing drop-connection"""
    def __init__(self, in_feat1, out_feat1, adj1, in_feat2, out_feat2, adj2):
        super(DropConnect, self).__init__()
        self.in_feat1 = in_feat1
        self.out_feat1 = out_feat1
        self.adj1 = adj1
        self.in_feat2 = in_feat2
        self.out_feat2 = out_feat2
        self.adj2 = adj2
        self.weight1 = nn.Parameter(torch.Tensor(out_feat1, in_feat1))
        self.bias1 = nn.Parameter(torch.Tensor(out_feat1))
        self.weight2 = nn.Parameter(torch.Tensor(out_feat2, in_feat2))
        self.bias2 = nn.Parameter(torch.Tensor(out_feat2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1)
        nn.init.kaiming_uniform_(self.weight2)

        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.bias1, -bound1, bound1)

        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, input1, input2):
        out1 = input1.matmul(self.weight1.t() * self.adj1) + self.bias1
        out2 = input2.matmul(self.weight2.t() * self.adj2) + self.bias2
        return torch.add(out1, out2)

    def get_indiv_outputs(self, input1, input2):
        """ Function to fetch individual output values """
        out1 = input1.matmul(self.weight1.t() * self.adj1) + self.bias1
        out2 = input2.matmul(self.weight2.t() * self.adj2) + self.bias2
        return out1, out2

    def extra_repr(self):
        return 'in_feat1={}, out_feat1={}, in_feat2={}, out_feat2={},'.format(
            self.in_feat1, self.out_feat1, self.in_feat2, self.out_feat2)


class DeepGami(nn.Module):
    """ DeepGAMI - Deep Auxiliary learning model """
    def __init__(self, n_feat1, adj1, n_feat2, adj2, latent_dim, model_type, nfc, dkp, n_classes):
        super(DeepGami, self).__init__()
        self.drop_conn = None
        self.fcn1, self.fcn2 = None, None
        self.n_out = n_classes
        if model_type == 'drop_connect':
            self.drop_conn = DropConnect(n_feat1, latent_dim, adj1, n_feat2, latent_dim, adj2)
        elif model_type == 'fully_connect':
            self.fcn1 = nn.Linear(n_feat1, latent_dim)
            self.fcn2 = nn.Linear(n_feat2, latent_dim)

        num_dc_out = 2*latent_dim
        self.ffn = FNN(num_dc_out, nfc, dkp)
        self.pred = nn.Linear(nfc[-1], n_classes)
        self.dropout = nn.Dropout(1-dkp)
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.alpha)
        if self.beta is not None:
            nn.init.constant_(self.beta, 0.1)

    def forward(self, inp1, inp2, estimate):
        if self.fcn1 is not None:
            ls_mod1 = self.fcn1(inp1)
            ls_mod2 = self.fcn2(inp2)
            ls_comb_mod = torch.add(ls_mod1, ls_mod2)
        else:
            ls_comb_mod = self.drop_conn(inp1, inp2)
            ls_mod1, ls_mod2 = self.drop_conn.get_indiv_outputs(inp1, inp2)

        ls_mod2_est = (self.alpha * ls_mod1) + self.beta
        ls_mod1_est = (ls_mod2 - self.alpha)/self.beta

        ls_mod1_out, ls_mod2_out = 0, 0
        if estimate == 'None':
            ls_mod1_out, ls_mod2_out = ls_mod1, ls_mod2
        elif estimate == 'cg':
            ls_mod1_out = ls_mod1
            ls_mod2_out = ls_mod2_est
        elif estimate == 'cs':
            ls_mod1_out = ls_mod1_est
            ls_mod2_out = ls_mod2

        ls_mod1_out = torch.add(ls_mod1_out, ls_comb_mod)
        ls_mod2_out = torch.add(ls_mod2_out, ls_comb_mod)
        ls_out = torch.cat([ls_mod1_out, ls_mod2_out], dim=1)
        ls_out = self.dropout(ls_out.relu())
        fc_out = self.ffn(ls_out)
        pred = self.pred(fc_out)
        if self.n_out == 1:
            return pred.sigmoid()
        else:
            return pred

    def get_intermediate_layers(self, inp1, inp2):
        """ Funciton to fetch the intermediate layer outputs """
        if self.fcn1 is not None:
            ls_mod1 = self.fcn1(inp1)
            ls_mod2 = self.fcn2(inp2)
        else:
            ls_mod1, ls_mod2 = self.drop_conn.get_indiv_outputs(inp1, inp2)

        ls_mod2_est = (self.alpha * ls_mod1) + self.beta
        ls_mod1_est = (ls_mod2 - self.alpha)/self.beta
        return ls_mod1, ls_mod1_est, ls_mod2, ls_mod2_est
