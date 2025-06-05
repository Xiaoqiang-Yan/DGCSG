# DGCSG
DGCSG: Differentiable Graph Clustering with Structural Grouping for Single-cell RNA-seq Data.

## Description
preprocess.py: preprocessing of raw scRNA-seq data.

ae_pretrain.py: pre-training on scRNA-seq data.

DGCSG.py: our model

main.py: train our model

## Environment Requirement
python==3.9.16

pytorch==1.12.0

numpy==1.22.0

sklearn==1.2.1

munkres==1.1.4

scipy==1.10.0

## Preprocess 
For example, put the raw expression matrix `ori_data.tsv` and the true label `label.ann` of the dataset Pollen into `/data/Pollen`, and run the following command to preprocess before clustering:
```python
python preprocess.py Pollen
```

## Run Code
For example, for the Pollen dataset, you can run the following command:
```python
python main.py --name Pollen
```

