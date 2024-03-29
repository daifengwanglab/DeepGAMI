# DeepGAMI - Deep auxiliary learning for multi-modal integration and estimation to improve genotype-phenotype prediction

Genotype-phenotype association is found in many biological systems such as brains and brain diseases. However, predicting phenotypes from genotypes remains challenging, primarily due to complex underlying molecular and cellular mechanisms. Emerging multi-modal data enables studying such mechanisms at different scales. However, most of these approaches fail to incorporate biology into the machine learning models. Due to the black-box nature of many machine learning techniques, it is challenging to integrate these multi-modalities and interpret the results for biological insights, especially when some modality is missing. To this end, we developed DeepGAMI, an interpretable deep learning model to improve genotype-phenotype prediction from multi-modal data. DeepGAMI uses prior biological knowledge to define the neural network architecture. Notably, it embeds an auxiliary-learning layer for cross-modal imputation while training the model from multi-modal data. Using this pre-trained layer, we can impute latent features of additional modalities and thus enable predicting phenotypes from a single modality only. Finally, the model uses integrated gradient approach to prioritize multi-modal features and links for phenotypes. 

![figure1_new](https://user-images.githubusercontent.com/57010174/169340516-1d3c46bb-9a4a-4d6a-a710-eeb168e8bb22.png)

DeepGAMI provides has three major steps:
* Training: Training phase involves training the data using 5-fold CV by tuning the hyperparameters (DeepGamiTrain.py).
* Testing: This step provides functions to predict new samples based on the trained model. We provide options to test when 1) Both modalities are present ((DeepGamiTest_SM.py)). 2) Single modality is present ((DeepGamiTest_DM.py)).
* Feature prioritization: This step uses the trained model to priotize input features associated with the phenotype ((DeepGamiIG.py)).

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

- Intermediate biological prior files for input 1 and input 2 (.csv files): These files represent prior biological knowledge that DeepGAMI requires for training using the biological dropconnect hidden layer. These intermediate files must contain "source" and "target" columns.  Additional columns like weight and other information can be provided as well but are optional. eQTLS and GRNs are examples of biological prior files. It is important to note that the feature set of input modalities must match the ‘source column’ of these files. Sometimes, these biological priors might be available. In these cases, you can specify None to convey the non-availability of these files. For example,the following screenshots show formats for input modality and its corresponding intermediate biological prior files:

<p align="center" width="100%">
    <img  src="https://github.com/daifengwanglab/DeepGAMI/blob/main/deepGAMI_inp1_format.png" >
    &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;
    <img  src="https://github.com/daifengwanglab/DeepGAMI/blob/main/deepGAMI_intermediate_bio_layer.png" >
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
* **--intermediate_phenotype_files** = This parameter specifies file path to input transparent layer adjacency matrix containing biological insights. (Default is None)
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

### Testing on new samples
We have provided code for the users to test our model on new samples. There are two versions of this test file:
* **DeepGamiTest_DM.py**: This is the traditional test classification file when both the input modalities are present. It can be run using the following command:
```
python -u DeepGamiTest_DM.py --input_files='/path_to_modality1_csv_file,/path_to_modality2_csv_file' --label_file='path_to_class_labels_csv_file' --model_file='/path_to_save_model' --task='binary' > '/path_to_output.txt'
```
* **DeepGamiTest_SM.py**: This file classifies new samples when only single modality is given as input. The other modality is imputed and then used for classification. It can be run using the following command:
```
python -u DeepGamiTest_SM.py --input_file='/path_to_modality1_csv_file' --label_file='path_to_class_labels_csv_file' --model_file='/path_to_trained_model' --task='binary'> '/path_to_output.txt'
```
The task can be binary or multiclass classification. Hence, we have provided an arguement 'task' to specify binary or multiclass.

The above two commands run the default settings for testing. Settings that can be included along with the above code are:
* **--input_file (for DeepGamiTest_SM.py)** = This parameter is used to specify input file path names for single modality.
* **--input_files (for DeeGamiTest_DM.py)** = This parameter is used to specify comma-separated input file path names for modalities.
* **--model_file** = This argument specifies the path to the tranied model generated by DeepGamiTrain.py file.
* **--label_file (Optional)** = This parameter is used to spcify path to the class labels of the samples used for testing if available (default = None).
* **--task** = This argument is used to spcify if the classification task is binary or multiclass (default = binary).

### Feature and Link Prioritization
DeepGami can also piriotize features and links using IntegratedGradient approach. For extracting the importance scores of the features and links, use DeepGamiIG.py file. The following command will help:
```
python -u DeepGamiIG.py --input_files='/path_to_modality1_csv_file,/path_to_modality2_csv_file' --label_file='path_to_class_labels_csv_file' --model_file='/path_to_trained_model' --prioritization_task='feature'> '/path_to_output.txt'
```

The above command runs the default settings for prioritization. Settings that can be modified along with the above code are:
* **--input_files** = This parameter is used to specify comma-separated input file path names for modalities.
* **--label_file** = This parameter is used to spcify path to the class labels of the samples used for testing if available.
* **--model_file** = This argument specifies the path to the tranied model generated by DeepGamiTrain.py file.
* **--prioritization_task** = This argument allows the user to specify feature, layer, or link prioritization (default = binary).

## Demo for predicting Cortical layers in single-cell Mouse Visual Cortex
This demo applies **DeepGami** to predict the cortical layers (L1,L2/L3,L4,L5,L6) for single-cell mutli-modal data from mouse visual cortex. The modalities provided as input include gene expression and electrophysiological features. DeepGami performs "standard" normalization by default, hence the input data can be raw files. Alternatively, you can choose to apply "minmax" normalization by setting the '--norm_type' parameter.

### Training
To train the model, run the following command:

```
python -u DeepGamiTrain.py --input_files "./demo/expMat_filtered.csv,./demo/efeature_filtered.csv" --disease_label_file "./demo/label_visual.csv" --num_fc_neurons '50' --latent_dim 100 --n_iter 100 --batch_size 30 --learn_rate 0.001 --out_reg 0.005 --corr_reg 1 --epochs 100 --cross_validate='True' --model_type='fully_connect' --save "." > "sc_MVC_result.txt"
```
The model generated by DeepGami is saved as "run_<*highest_acc_epoch_number*>\_bestmodel.pth". For the above command, the following files are generated:
* "run_92_best_model.pth" - The trained model
* sc_MVC_result.txt - log file
* overall_perf.txt - This files gives you the performance of DeepGami. It contains balanced accuracy and AUC scores for training, dual-modality validation, and single-modality validation.
* tr_2m_perf.txt - Contains training balanced accuracies for each phenotype class.
* val_2m_perf.txt - Contains dual-modality validation balanced accuracies for each phenotype class.
* val_1m_cg_perf.txt - Contains single-modality validation balanced accuracies for each phenotype class.

**Note:** The above modalities were not trained with intermediate biological layers, hence the model_type is set to 'fully_connected' and a latent_dim1=100 is provided. 

### Testing
The trained model can then be used to predict the labels on new samples. **DeepGamiTest_DM.py** is used when both data modalities are available and **DeepGamiTest_SM.py** is used when only one modality is present. Here, DeepGamiTest_SM.py is used to demonstrate predicting new samples using single modality with the assumption that the class labels are not available. The following command will predict new samples:

```
python -u DeepGamiTest_SM.py --input_file='./demo/test/independent_test_118_gexMat.csv' --model_file='./demo/run_92_best_model.pth' --task='multiclass'
```

This code generates the following output:
<p align="center" width="100%">
    <img  src="https://github.com/daifengwanglab/DeepGAMI/blob/main/sample_output.png" >
</p>



#### Feature Prioritization
DeepGamiIG.py can be used to priotiritze electrophysiological and gene features. Run the command:
```
python -u DeepGamiIG.py --input_files='demo/expMat_filtered.csv,demo/efeature_filtered.csv' --label_file='demo/label_visual.csv' --model_file='run_92_best_model.pth' --prioritization_task='feature'
```

This generates two files: "mod1_prioritized_link.csv" containing prioritized gene features and its importance score and "mod2_prioritized_link.csv" for electrophysiological prioritization with its importance score.

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
