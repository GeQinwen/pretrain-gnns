
## Guide for Reproducing the results

1. 
To reproduce the first part of replication study, first need to dowload the chemistry dataset below:
 [chem data](https://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `chem/`.
To reproduce the result, setting up a conda environment is recommended since some libraries need to be old version:
```
Python 3.7
pytorch                   1.0.1
torch-cluster             1.2.4              
torch-geometric           1.0.3
torch-scatter             1.1.2 
torch-sparse              0.2.4
torch-spline-conv         1.0.6
rdkit                     2019.03.1.0
tqdm                      4.31.1
tensorboardx              1.6
```
Next step is to just run the following line in terminal under the `chem/`, also the same dictionary of `finetune_tune.sh`
sh finetune_tune.sh SEED DEVICE

2. 
Other results which evaluates the pretraining strategy on broader domains can be found in the `pretrain.ipynb`.
It was run on new version of all libraries instead of previous environment.
Some cells could be pretty slow.

# Strategies for Pre-training Graph Neural Networks

This is a Pytorch implementation of the following paper: 

Weihua Hu*, Bowen Liu*, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec. Strategies for Pre-training Graph Neural Networks. ICLR 2020.
[arXiv](https://arxiv.org/abs/1905.12265) [OpenReview](https://openreview.net/forum?id=HJlWWJSFDH) 

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).

```
@inproceedings{
hu2020pretraining,
title={Strategies for Pre-training Graph Neural Networks},
author={Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJlWWJSFDH},
}
```



