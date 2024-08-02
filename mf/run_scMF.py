import scanpy as sc
from time import time
import math, os
import torch
from functools import reduce
from scMF import MF
import numpy as np
import collections
import h5py
from preprocess import read_dataset, normalize
from utils import *

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--data_file', default='./realdata/10X_PBMC_newCount_filtered_1000G.H5')
    parser.add_argument('--total_epochs', default=1000, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--weight_file', default='AE_weights.pth.tar')
    parser.add_argument('--output_file', default='pred.csv')
    args = parser.parse_args()
    print(args)
    
    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    b = np.array(data_mat['B'])
    w = np.array(data_mat['W'])
    w = w.astype(np.float)
    data_mat.close()
    
    #check zero
    zero = reduce(np.union1d,(np.where(np.sum(x,axis=1)==0), np.where(np.sum(b,axis=1)==0)))
    x = np.delete(x,zero, axis=0)
    b = np.delete(b,zero, axis=0)
    w = np.delete(w,zero, axis=0)
    zero1 = np.where(np.sum(x,axis=0)==0)
    x = np.delete(x,zero1, axis=1)
    w = np.delete(w,zero1, axis=1)
    print(x.shape)
    print(b.shape)
    print(w.shape)

    # preprocessing scRNA-seq read counts matrix
    adata1 = sc.AnnData(x)

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    adata2 = sc.AnnData(b)
    adata2 = clr_normalize_each_cell(adata2)

    print(adata1.X.shape)
    print(adata2.X.shape)
     
    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars

    model = MF(input_dim=input_size1, z_dim=input_size2, w=w).cuda()
    
    print(model)
    
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir) 
    
    model.fit(X=adata1.X, B=adata2.X, batch_size=args.batch_size, epochs=args.total_epochs, ae_weights=args.save_dir+args.weight_file)
    
    print("Done")
    
    weights = torch.load("results/AE_weights.pth.tar",map_location=torch.device('cpu'))
    out = weights['ae_state_dict']['encoder.0.weight'].numpy().T
    np.savetxt(args.save_dir + args.output_file, out, delimiter=",")
    
    keep_genes = adata1.var_names
    keep_genes = np.array(keep_genes.astype(np.int))
    np.savetxt(args.save_dir + "keepgenes.csv", keep_genes, delimiter=",")
