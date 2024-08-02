from time import time
import math, os

from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from functools import reduce
from scMultiCluster_kl_DEC import scMultiCluster
import numpy as np
import pandas as pd
import collections
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
from utils import *

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--cutoff1', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--cutoff2', default=0., type=float, help='Start to train combined layer after what ratio of batch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='./Normalized_filtered_10X_pbmc_granulocyte_plus.h5')
    parser.add_argument('--maxiter', default=800, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--beta', default=.1, type=float,
                        help='coefficient of latent autoencoder loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--hcae_weight_file', default='HCAE_weights_1.pth.tar')
    parser.add_argument('--embedding_file', default=1)
    parser.add_argument('--prediction_file', default=1)
    parser.add_argument('-l1','--encodeLayer1', nargs='+', default=[256,128,64])
    parser.add_argument('-l2','--encodeLayer2', nargs='+', default=[256,128,64])
    parser.add_argument('-ll','--encodeLayerLatent', nargs='+', default=[64,32])
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=2.5, type=float)
    parser.add_argument('--filter', default=1, type=int)
    parser.add_argument('--f1', default=2000, type=float)
    parser.add_argument('--f2', default=2000, type=float)
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--ml_weight', default=.1, type=float)
    parser.add_argument('--cl_weight', default=1., type=float)
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    
    data_mat = h5py.File(args.data_file)
    x1 = np.array(data_mat['X1'])
    x2 = np.array(data_mat['X4'])
    adt = np.array(data_mat['X2'])
    y = np.array(data_mat['Y'])
    data_mat.close()
    print(x1.shape)
    print(x2.shape)
    print(adt.shape)
    print(y.shape)
    
    #Gene filter
    if args.filter != -1:
        print("Doing gene filtering")
        importantGenes = geneSelection(x1, n=args.f1, plot=False)
        x1 = x1[:, importantGenes]
    
        importantGenes = geneSelection(x2, n=args.f2, plot=False)
        x2 = x2[:, importantGenes]
    
    #check zero
    zero1 = reduce(np.union1d,(np.where(np.sum(x1,axis=1)==0), np.where(np.sum(x2,axis=1)==0),np.where(np.sum(adt,axis=1)==0)))
    print(zero1)
    x1 = np.delete(x1,zero1, axis=0)
    x2 = np.delete(x2,zero1, axis=0)
    adt = np.delete(adt,zero1, axis=0)
    y = np.delete(y,zero1)
    np.savetxt(args.save_dir + "/" + str(args.batch) +  "_cellfilter.csv", zero1, delimiter=",")
    
    print(x1.shape)
    print(x2.shape)
    print(adt.shape)
    print(y.shape)
    
    #build constraints
    adata0 = sc.AnnData(adt)
    adt = clr_normalize_each_cell(adata0)
    print(adt.shape)

    adt = adt[:, 4]
    label_cell_indx = np.arange(len(y))
    low = np.quantile(adt,0.25)
    high = np.quantile(adt,0.75)
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(adt, label_cell_indx, low, high, num1=4000, num2=4000)

    # preprocessing scRNA-seq read counts matrix
    adata1 = sc.AnnData(x1)
    adata1.obs['Group'] = y

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    adata2 = sc.AnnData(x2)
    adata2.obs['Group'] = y
    adata2 = read_dataset(adata2,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata2 = normalize(adata2,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars
    
    print(args)
    
    encodeLayer1 = list(map(int, args.encodeLayer1))
    decodeLayer1 = encodeLayer1[::-1]
    encodeLayer2 = list(map(int, args.encodeLayer2))
    decodeLayer2 = encodeLayer2[::-1]
    encodeLayerLatent = list(map(int, args.encodeLayerLatent))
    if len(encodeLayerLatent) >1:
       decodeLayerLatent = encodeLayerLatent[::-1]
    else:
       decodeLayerLatent = encodeLayerLatent
    
    model = scMultiCluster(input_dim1=input_size1, input_dim2=input_size2, n_clusters = args.n_clusters, 
                        zencode_dim=encodeLayerLatent, zdecode_dim=decodeLayerLatent, 
                        encodeLayer1=encodeLayer1, decodeLayer1=decodeLayer1, encodeLayer2=encodeLayer2, decodeLayer2=decodeLayer2,
                        sigma1=args.sigma1, sigma2=args.sigma2, beta=args.beta, gamma=args.gamma, ml_weight=args.ml_weight, cutoff1 = args.cutoff1, cutoff2 = args.cutoff2).cuda()
    
    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))          

    y_pred, _, _, _, _ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, y=y, batch_size=args.batch_size, 
                                   ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2, ml_p = args.ml_weight, cl_p = args.cl_weight,
                                   num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, lr=args.lr, save_dir=args.save_dir, hcae_weights=args.hcae_weight_file)
    print('Total time: %d seconds.' % int(time() - t0))
    
    if args.prediction_file != -1:
       np.savetxt(args.save_dir + "/" + str(args.batch) + "_pred.csv", y_pred, delimiter=",")
    
    if args.embedding_file != -1:
       final_latent,_,_ = model.encodeBatch(torch.tensor(adata1.X).cuda(), torch.tensor(adata2.X).cuda())
       final_latent = final_latent.cpu().numpy()
       np.savetxt(args.save_dir + "/" + str(args.batch) +  "_embedding.csv", final_latent, delimiter=",")

    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    ami = np.round(metrics.adjusted_mutual_info_score(y, y_pred), 5)
    print('Final: NMI= %.4f, ARI= %.4f, AMI=%.4f' % (nmi, ari, ami))
    print(np.unique(y_pred))
