#!/bin/bash
# ROSMAP braaksc
# modality1: chipseq; modality2: gex


python3 -u DeepGamiTrain.py --input_files="./../rosmap_GexEpi_noFS_latest/rosmap_GexEpi_chip_noFS_tr.csv,./../rosmap_GexEpi_noFS_latest/rosmap_GexEpi_gex_noFS_tr.csv" --intermediate_phenotype_files="./../rosmap_GexEpi_noFS_latest/rosmap_GexEpi_epl_adj_noFS_new.npz,./../rosmap_GexEpi_noFS_latest/rosmap_GexEpi_grn_adj_noFS_new.npz" --disease_label_file="./../rosmap_GexEpi_noFS_latest/rosmap_GexEpi_phen_noFS_tr.csv" --model_type='drop_connect' --cross_validate True --num_fc_neurons='350' --save "./deepgami_models/rosmap_cogdx_nfc350_split/" > "./results/deepgami_cogdx_nfc350_split.txt"

