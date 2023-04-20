#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pramod

 - Modify lines 19-27 to point to the locations of the input files and cell type.
 - Modify lines 120, 123, 126, 129, 134, 140, 143,146, 147 
   to point to the dir where the output files are to be written.
"""

import pandas as pd
import networkx as nx
import scipy as sp
import numpy as np
import pickle as pk


def readData(gex_file, snp_file, phen_file, grn_file, eqtl_file):
    """ Read the data form files"""
    # Read Gene Expression Data
    gex = pd.read_csv(gex_file).set_index('Unnamed: 0')
    gex = gex.T
    print('Input Modality 1', gex.shape)
    gex_samples = gex.columns.to_list()
    
    # Read SNP information
    snps = pd.read_csv(snp_file).set_index('Unnamed: 0')
    snps = snps.T
    print('Input Modality 2', snps.shape)
    snp_samples=snp.columns.to_list()
    
    # Read class labels
    phen = pd.read_csv(phen_file)
    phen.index=phen['individualID']
    
    common_samples = list(set(gex_samples).intersection(set(snp_samples)))
    common_samples = list(set(common_samples).intersection(set(phen_samples)))
    gex = gex[common_samples]
    snp = snp[common_samples]
    phen = phen.loc[common_samples] 
    
    # Read eqtl data
    if eqtl_file is not None and grn_file is not None:
        eqtl = pd.read_csv(eqtl_file).assign(weight=1)
        #eqtl = eqtl[['snp_id','gene_id','weight']]
        eqtl = eqtl[['source','target','weight']]
        eqtl.columns = ['source','target','weight']
        print('intermediate file:2', eqtl.shape)
        
        # Read GRN data
        grn = pd.read_csv(grn_file).assign(weight=1)
        grn = grn[['source','target','weight']]
        #grn = grn.drop(columns=['Edge_Weight'])
        grn.columns = ['source','target','weight']
        print('intermediate file:1', grn.shape)
        return gex, snps, phen, grn, eqtl
    else:
        return gex, snps, phen

def get_adj_mat(df, type='eqtl', genes=None):
    """ get adjacency matrix """
    G = nx.from_pandas_edgelist(df, create_using=nx.DiGraph())
    
    if type == 'eqtl':
        idx = df[['source','target']].stack().reset_index(level=[0], drop=True).drop_duplicates().reset_index()
        col_idx = idx[idx['index']=='target'].index.values
        row_idx = idx[idx['index']=='source'].index.values
        
        target_ls = idx[idx['index']=='target'][0].tolist()
        source_ls = idx[idx['index']=='source'][0].tolist()
        adj = nx.adjacency_matrix(G)
        adj = sp.sparse.csr_matrix(adj.tocsr()[row_idx,:][:,col_idx].todense())
        adj_d = pd.DataFrame(adj.todense(), index=source_ls, columns=target_ls)
    else:
        node_ls = list(set(df.source).union(set(df.target)))
        adj = nx.adjacency_matrix(G, nodelist=node_ls)
        adj_d = pd.DataFrame(adj.todense(), index=node_ls, columns=node_ls)
    print(adj_d.shape)

    adj_final = adj_d.loc[:, adj_d.columns.isin(genes)]
    return adj_final

def match_samples_and_features(gex, snps, grn, eqtl):
    #gex, snps, grn, eqtl = cmc_tr_gex.copy(),cmc_tr_snps.copy(), grn.copy(), eqtl.copy()
    # Get the common gene list determining the hidden layer
    grn_tfs = list(set(grn.source.values))
    grn_tgs = list(set(grn.target.values))
    grn_genes = list(set(grn_tfs).union(set(grn_tgs)))
    eqtl_genes = list(set(eqtl.target.values))
    gex_genes = list(gex.index.values)

    #print('GexGenes n eqtlGenes', len(set(gex_genes).intersection(set(eqtl_genes))))
    #print('GexGenes n grnGenes', len(set(gex_genes).intersection(set(grn_genes))))
    #print('grnGenes n eqtlGenes', len(set(eqtl_genes).intersection(set(grn_genes))))
    #print('GexGenes n grnTFs', len(set(gex_genes).intersection(set(grn_tfs))))
    #print('GexGenes n grnTGs', len(set(gex_genes).intersection(set(grn_tgs))))
    #print('eqtl_genes n grnTGs', len(set(eqtl_genes).intersection(set(grn_tgs))))

    comm_genes = list(set(eqtl_genes).intersection(set(grn_tgs)))

    # Filter eqtls and grn target genes based on the common genes
    eqtl = eqtl.loc[eqtl.target.isin(comm_genes), :].reset_index(drop=True)
    grn = grn.loc[grn.target.isin(comm_genes), :].reset_index(drop=True)


    # Eqtl adjacency matrix
    eqtl_adj_final = get_adj_mat(eqtl, type='eqtl', genes=comm_genes)
    #print('eqtl_adj_final', eqtl_adj_final.shape)

    # GRN adjacency matrix
    grn_adj_final = get_adj_mat(grn, type='grn', genes=comm_genes)
    #print('grn_adj_v1', grn_adj_final.shape)
    grn_adj_final = grn_adj_final.loc[grn_adj_final.index.isin(grn_tfs), :]
    grn_adj_final = grn_adj_final.loc[:, grn_adj_final.columns.isin(comm_genes)]
    #print('grn_adj_final', grn_adj_final.shape)

    if eqtl_adj_final.shape[1] > grn_adj_final.shape[1]:
        eqtl_adj_final = eqtl_adj_final.loc[:, eqtl_adj_final.columns.isin(grn_adj_final.columns)]
        nz_series = (eqtl_adj_final != 0).any(axis=1)
        eqtl_adj_final = eqtl_adj_final.loc[nz_series, :]
    #print('eqtl_adj_final', eqtl_adj_final.shape)

    eqtl_adj_final = eqtl_adj_final[sorted(eqtl_adj_final.columns)]
    grn_adj_final = grn_adj_final[sorted(grn_adj_final.columns)]

    # Get common genes between grn source and gene expression
    grn_gex_genes = list(set(gex_genes).intersection(set(grn_adj_final.index)))
    grn_adj_final = grn_adj_final.loc[grn_adj_final.index.isin(grn_gex_genes), :]
    gex = gex.loc[gex.index.isin(grn_gex_genes), :]


    # Filter snps based on eqtl snps
    comm_snps = list(set(eqtl_adj_final.index).intersection(snps.index))
    snps = snps.loc[snps.index.isin(comm_snps), :]
    eqtl_adj_final = eqtl_adj_final.loc[eqtl_adj_final.index.isin(comm_snps), :]

    # Sort gex and snps based on eqtl and grn
    gene_ls = list(grn_adj_final.index)
    snp_ls = list(eqtl_adj_final.index)
    gex = gex.reindex(gene_ls)
    snps = snps.reindex(snp_ls)
    
    print('preprocessed modality 1', gex.shape)
    print('preprocessed modality 2', snps.shape)
    print('preprocessed intermediate 1', grn_adj_final.shape)
    print('preprocessed intermediate 2', eqtl_adj_final.shape)
    
    return gex, snps, grn_adj_final, eqtl_adj_final


def save_files(gex, geno, phen, grn=None, eqtl=None, fldr_name='/results', file_prefix='cmc'):
    # write out gene names
    with open(fldr_name + file_prefix +'_tfs.list','w') as glf:
        for g in list(gex.index):
            glf.write(g+"\n") 
    
    # write snp list
    with open(fldr_name + file_prefix +'_snps.list','w') as glf:
        for g in list(geno.index):
            glf.write(g+"\n") 
    
    # write target genes list
    with open(fldr_name + file_prefix +'_tgs.list','w') as glf:
        for g in list(eqtl.columns):
            glf.write(g+"\n") 
    
    
    # Write each individual data types
    gex.to_csv(fldr_name + file_prefix +'_modality1.csv')
    geno.to_csv(fldr_name + file_prefix +'_modality2.csv')
      
    if eqtl is not None and grn is not None:
        eqtl_adj_sparse_mat = sp.sparse.csr_matrix(eqtl)
        grn_adj_sparse_mat = sp.sparse.csr_matrix(grn)

        sp.sparse.save_npz(fldr_name + file_prefix +'_modality2_adj.npz', eqtl_adj_sparse_mat)
        sp.sparse.save_npz(fldr_name + file_prefix +'_modality1_adj.npz', grn_adj_sparse_mat)
    
    phen.to_csv(fldr_name + file_prefix +'_labels.csv')    

# ------------------ Analysis ------------------------------
def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_1', type=str,
                        help='Path to the input modality 1 data file')
    parser.add_argument('--input_file_2', type=str,
                        help='Path to the input modality 2 data file')
    parser.add_argument('--intermediate_biological_file_1', type=str,
                        default="None",
                        help='Path to transparent layer adjacency matrix for modality 1')
    parser.add_argument('--intermediate_biological_file_2', type=str,
                        default="None",
                        help='Path to transparent layer adjacency matrix for modality 2')
    parser.add_argument('--disease_label_file', type=str,
                        help='Path to Output labels file - Disease phenotypes')
                        
    args = parser.parse_args()
    gex_file=args.input_file_1
    snp_file=args.input_file_2
    grn_file=args.intermediate_phenotype_file_1
    eqtl_file=args.intermediate_phenotype_file_2
    phen_file=args.disease_label_file
    
    cmc_tr_gex, cmc_tr_snps, cmc_tr_phen, grn, eqtl = readData(gex_file, snp_file, phen_file, grn_file, eqtl_file)
    grn = grn.drop_duplicates().reset_index(drop=True)

    if ge_file is not None and eqtl_file is not None:
        gex_flt, snps_flt, grn_adj, eqtl_adj = match_samples_and_features(cmc_tr_gex.copy(),
		                                                          cmc_tr_snps.copy(),
		                                                          grn.copy(), eqtl.copy())
    else:
        gex_flt=cmc_tr_gex
        snps_flt= cmc_tr_snps
        cmc_tr_phen= cmc_tr_phen
        grn_adj= None
        eqtl_adj= None
        
    save_files(gex_flt, snps_flt, cmc_tr_phen, grn_adj, eqtl_adj, fldr_name='./preprocessed_data/', file_prefix='preprocessed')


if __name__ == '__main__':
    main()





