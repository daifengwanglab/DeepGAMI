#!/bin/bash

# train the model
python -u DeepGamiTrain.py --input_files "./demo/expMat_filtered.csv,./demo/efeature_filtered.csv" --disease_label_file "./demo/label_visual.csv" --num_fc_neurons '50' --latent_dim 100 --n_iter 100 --batch_size 30 --learn_rate 0.001 --out_reg 0.005 --corr_reg 1 --epochs 100 --cross_validate='True' --model_type='fully_connect' --save "." > "sc_MVC_result.txt"

