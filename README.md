# Multi-SC

## Description
Our pipeline leverages a multimodal constraint autoencoder (scHCAE) to integrate the multiomics data during the clustering process and a matrix factorization-based model (scMF) to predict target genes regulated by a TF.

## Usage
### Prerequisites
- Python 3.8 or higher
- Required packages: `torch`, `sklearn`, `scipy`, `scanpy`, `h5py`, `numpys`, `pandas`

### Setup
```bash
git clone https://github.com/xianglin226/Multi-SC.git
```

### Running scHCAE
To run the scHCAE for clustering with a specified number of clusters, use the following command:
```bash
python -u run_scMultiCluster3.py \
 --n_clusters 6 \
 --data_file GSE178707_neatseq_lane1.h5
```

### Running scMF
To predict target genes regulated by TFs using scMF, run:
```bash
python -u run_scMF.py \
 --data_file processedinput_scMF_lane1.h5
```

### Example Data
The example data can be access [here](https://drive.google.com/drive/folders/1Sq0w03-tFc-fcv-Y1i9XmL4TIxkoLCKq?usp=share_link).  


