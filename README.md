# DeepGAMI - Deep auxiliary learning for multi-modal integration and estimation to improve genotype-phenotype prediction

Genotype-phenotype association is found in many biological systems such as brains and brain diseases. However, predicting phenotypes from genotypes remains challenging, primarily due to complex underlying molecular and cellular mechanisms. Emerging multi-modal data enables studying such mechanisms at different scales. However, most of these approaches fail to incorporate biology into the machine learning models. Due to the black-box nature of many machine learning techniques, it is challenging to integrate these multi-modalities and interpret the results for biological insights, especially when some modality is missing. 

To this end, we developed DeepGAMI, an interpretable deep learning model to improve genotype-phenotype prediction from multi-modal data. DeepGAMI uses prior biological knowledge to define the neural network architecture. Notably, it embeds an auxiliary-learning layer for cross-modal imputation while training the model from multi-modal data. Using this pre-trained layer, we can impute latent features of additional modalities and thus enable predicting phenotypes from a single modality only. Finally, the model uses integrated gradient approach to prioritize multi-modal features and links for phenotypes. We applied DeepGAMI to (1) population-level bulk and cell-type-specific genotype and gene expression data for Schizophrenia (SCZ) cohort, (2) genotype and gene expression data for Alzheimer's Disease (AD) cohort, and (3) recent single-cell multi-modal data comprising transcriptomics and electrophysiology for neuronal cells in the mouse visual cortex. We found that DeepGAMI outperforms existing state-of-the-art methods and provides a profound understanding of gene regulatory mechanisms at cellular resolution from genotype to phenotype. 

![figure1_new](https://user-images.githubusercontent.com/57010174/169340516-1d3c46bb-9a4a-4d6a-a710-eeb168e8bb22.png)



## Dependencies
The script is based on python 3.4 above and requires the following packages:
- pytorch: v1.4.0  (cpu) or v1.10.0(gpu)
- scipy
- numpy
- scikit-learn
- pandas
- captum
- imblearn

## Download code
```python
git clone https://github.com/daifengwanglab/DeepGAMI
cd DeepGAMI
```

## Usage

### Input files
To train the DeepGAMI model, users are required to provide
- Input data modalities (input_1.csv and input_2.csv): These files are the main input to the model. They must be in .csv format and should contain rows as samples/cells and columns as features, such as genes or SNPs. For example Input 1: Genotype (samples as rows and SNPs as columns. The value represents either dosage or genotype) and Input 2: Gene expression (samples as rows and genes as columns).

- Intermediate biological prior files for input 1 and input 2 (.csv files): These files represent prior biological knowledge that DeepGAMI requires for training using the biological dropconnect hidden layer. These intermediate files must contain "source" and "target" columns.  Additional columns like weight and other information can be provided as well but are optional. eQTLS and GRNs are examples of biological prior files. It is important to note that the feature set of input modalities must match the ‘source column’ of these files. Sometimes, these biological priors might be available. In these cases, you can specify None to convey the non-availability of these files. For example,


<p align="center" width="100%">
    <img  src="https://github.com/daifengwanglab/DeepGAMI/blob/main/deepGAMI_inp1_format.png" title="Input modality 1">
    <img  src="https://github.com/daifengwanglab/DeepGAMI/blob/main/deepGAMI_intermediate_bio_layer.png" title="Intermediate Biological prior file for modality 1">
</p>


- Disease phenotype file (.csv file): This file should contain the labels for training the samples/cells in input modalities The labels column must be marked as "labels", and the sample/cell IDs as "individualID".

### Preprocess Data
This step formats the input modalities, any intermediate biological layer files, and the phenotype file to make sure that features and sample IDs across modalities and labels are aligned. 
```
python -u data_preprocess.py --input_file_1='/path_to_modality1_csv_file' --input_file_2='/path_to_modality2_csv_file' --intermediate_biological_file_1= '/path_to_intermediate_biological_file_1' --intermediate_biological_file_2='/path_to_intermediate_biological_file_2' --disease_label_file='/path_to_class_labels_csv_file' 
```
The above command generates the following output files to train the DeepGAMI model:
- preprocessed_modality1.csv, preprocessed_modality2.csv: files contain modalities with features matched with their corresponding intermediate biological layer features (*Use these as input modalities for training*)
- preprocessed_modality1_adj.npz, preprocessed_modality2_adj.npz: files containing adjacency matrices with features matched with their corresponding modalities (*Use these as intermediate biological files for training*)
- preprocessed_labels.csv: file contains phenotypes/labels aligned according to sample IDs in modalities (*Use this as disease phenotype file for training*)
- preprocessed_tfs.list: file contains features for modality 1
- preprocessed_snps.list: file contains features for modality 2
- preprocessed_tgs.list: file contains intermediate features for modality 1

### Train DeepGAMI model
The DeepGamiTrain file is the starting point which reads the input files, performs model training, and predicts the phenotype. It can be run using the following command:

```
python -u DeepGamiTrain.py --input_files='/path_to_modality1_csv_file,/path_to_modality2_csv_file' --disease_label_file='path_to_class_labels_csv_file' --save= '/path_to_save_model' > '/path_to_output.txt'
```
The above code uses fully connected network for connecting the input to the intermediate transparent layer. For guiding the model using biological insights, use the following command:

```
python -u DeepGamiTrain.py --input_files='/path_to_modality1_csv_file,/path_to_modality2_csv_file' --intermediate_phenotype_files='/path_to_biological_insights_adjacency_matrix1,/path_to_biological_insights_adjacency_matrix2' --disease_label_file='path_to_class_labels_csv_file' --model_type='drop_connect' --save= '/path_to_save_model' > '/path_to_output.txt'
```

The above two command runs the default settings for training. Additional settings that can be included along with the above code are:
* **--num_data_modal** = Specify the number of modalities (default=2. Currently supports only 2 modalities).
* **--input_files** = This parameter is used to specify comma-separated input file path names for modalities.
* **--intermediate_biological_files** = This parameter specifies file path to input transparent layer adjacency matrix containing biological insights. (Default is None)
* **--disease_label_file** = This parameter specifies file path for output labels (eg. disease phenotypes).
* **--learn_rate** = Learning rate for the model. (default = 0.001) 
* **--out_reg** = L2 regularization parameter. (default = 0.005)
* **--corr_reg** = Regularization parameter for the cross-modal estimation loss. (default = 0.5)
* **--model_type** = This parameter is used to determine if the intermediate layer undergoes a fully connected network or biological dropconnection. (default='fully_connect')
* **--latent_dim** = This parameter is used to specify the number of hidden nodes in the transparent layer if the model type is fully conencted network. (default=100)
* **--num_fc_neurons** = Number of hidden units for fully connected layers. To add additional hidden layers, provide number of hidden units for each layer separated by a comma for e.g. '200,100,50' has 3 layers with 200, 100 and 50 hidden units respectively. (default = '500,50')
* **--dropout_keep_prob** = This is used to handle overfitting. (default = 0.5)
* **--need_normalization** = Flag for perfroming data normalziation.
* **--norm_type** = Feature normalization versus sample normalization. (default = 'features')
* **--norm_method** = Specify normalization method (Standard/log/min-max/None) to use for each modality separated by comma for e.g. 'Standard,None' applies standard normalization to modality 1 and no normalization for modality 2.
* **--train_percent** = Choose how the tain and test validation split to occur. (default=0.8)
* **--need_balance** = This specifies balanced training. (default=False)
* **--batch_size** = Batch size for training. (default = 30)
* **--epochs** = Number of epochs. (default=100)
* **--stagnant** = Specify the early stop criterion i.e. the number of iterations after which to stop. (default=100)
* **--n_iter** = Specify the number of iterations. (default=1)
* **--oversampling** = Flag to perfrom oversampling based on the unbalanced data. (default=False)
* **--cross-validate** = This is a flag which performs 5-fold CV when enabled. (default=True)
* **--save** = This argument specifies the path to save the model generated by the code.

**Note:** If the intermediate biological files are not provided, then users must specify model_type as 'fully_connect' and a value for latent_dim. 

## Demo for predicting Cortical layers in single-cell Mouse Visual Cortex
This demo applies **DeepGami** to predict the cortical layers (L1,L2/L3,L4,L5,L6) for single-cell mutli-modal data from mouse visual cortex. The modalities provided as input include gene expression and electrophysiological features. DeepGami performs "standard" normalization by default, hence the input data can be raw files. Alternatively, you can choose to apply "minmax" normalization by setting the '--norm_type' parameter.

### Training
To train the model, run the following command:

```
python -u DeepGamiTrain.py --input_files "./demo/expMat_filtered.csv,./demo/efeature_filtered.csv" --disease_label_file "./demo/label_visual.csv" --num_fc_neurons '50' --latent_dim 100 --n_iter 100 --batch_size 30 --learn_rate 0.001 --out_reg 0.005 --corr_reg 1 --epochs 100 --cross_validate='True' --model_type='fully_connect' --save "." > "sc_MVC_result.txt"
```
The model generated by DeepDice is saved as "run_<*highest_acc_epoch_number*>\_bestmodel.pth". For the above command, the following files are generated:
* "run_<*highest_acc_epoch_number*>\_bestmodel.pth" - The trained model
* sc_MVC_result.txt - log file
* overall_perf.txt - This files gives you the performance of DeepGami. It contains balanced accuracy and AUC scores for training, dual-modality validation, and single-modality validation.
* tr_2m_perf.txt - Contains training balanced accuracies for each phenotype class.
* val_2m_perf.txt - Contains dual-modality validation balanced accuracies for each phenotype class.
* val_1m_cg_perf.txt - Contains single-modality validation balanced accuracies for each phenotype class.

**Note:** The above modalities were not trained with intermediate biological layers, hence the model_type is set to 'fully_connected' and a latent_dim1=100 is provided. 

#### Feature and Link Prioritization
To get feature and link prioritizations, DeepDice uses the fucntions: IntegratedGradients, LayerConductance and NeuronConductance from the captum package. In the "DeepGamiIG.py" file, specify the input files and the model filename (generated by DeepDice) as follows:

```
input_files = "demo/expMat_filtered.csv,demo/efeature_filtered.csv"
mid_phen_files = "None"
label_file = "demo/label_visual.csv"
model_file = "run_<*highest_acc_epoch_number*>\_bestmodel.pth"
```
Then run the command:
```
python DeepGamiIG.py
```
This generates two files: "ephys_prioritized.csv" containing prioritized electrophysiological features and its importance score, and "genes_prioritized.csv" for gene prioritization with its importance score.

## License
MIT License

Copyright (c) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
