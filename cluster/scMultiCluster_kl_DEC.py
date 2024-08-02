from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import NBLoss, ZINBLoss, MeanAct, DispAct
import numpy as np

import math, os

from utils import best_map
#from pytorch_kmeans import kmeans_torch, initialize, pairwise_distance, pairwise_cosine, kmeans_predict

from preprocess import read_dataset, normalize
import scanpy as sc

def buildNetwork1(layers, type, activation="elu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if type=="encode" and i==len(layers)-1:
            break
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

def buildNetwork2(layers, type, activation="elu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=False))
        if type=="encode" and i==len(layers)-1:
            break
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation=="selu":
            net.append(nn.SELU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

class scMultiCluster(nn.Module):
    def __init__(self, input_dim1, input_dim2, n_clusters = 7, zencode_dim=[64,16], zdecode_dim=[16,64],
            encodeLayer1=[256, 128, 64], decodeLayer1=[64, 128, 256], encodeLayer2=[256, 128, 64], decodeLayer2=[64, 128, 256],
            activation="elu", sigma1=2.5, sigma2=1., alpha=1., beta=1., gamma=1.,ml_weight=0.1, cutoff1 = 0.4, cutoff2 = 0.4):
        super(scMultiCluster, self).__init__()
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.z_dim = zencode_dim[-1]
        self.encoder1 = buildNetwork2([input_dim1]+encodeLayer1, type="encode", activation=activation)
        self.decoder1 = buildNetwork2(decodeLayer1, type="decode", activation=activation)
        self.encoder2 = buildNetwork2([input_dim2]+encodeLayer2, type="encode", activation=activation)
        self.decoder2 = buildNetwork2(decodeLayer2, type="decode", activation=activation)
        self.latent_enc = buildNetwork2([encodeLayer1[-1]+encodeLayer2[-1]]+ zencode_dim, type="encode", activation=activation)
        #self.latent_enc = buildNetwork([encodeLayer1[-1]+ input_dim2]+zencode_dim, type="encode", activation=activation)
        self.latent_dec = buildNetwork2(zdecode_dim+[encodeLayer1[-1]+encodeLayer2[-1]], type="encode", activation=activation)        
        #self.latent_dec = buildNetwork(zdecode_dim+[encodeLayer1[-1]+input_dim2], type="encode", activation=activation)
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()
        self.NBLoss = NBLoss()
        self.mse = nn.MSELoss()
        self.n_clusters = n_clusters
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
        
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p
     
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forward(self, x1, x2):
        h1 = self.encoder1(x1+torch.randn_like(x1) * self.sigma1)
        h2 = self.encoder2(x2+torch.randn_like(x2) * self.sigma2)
        
        h1_ = self.decoder1(h1)
        mean1 = self.dec_mean1(h1_)
        disp1 = self.dec_disp1(h1_)
        pi1 = self.dec_pi1(h1_)
        
        h2_ = self.decoder2(h2)
        mean2 = self.dec_mean2(h2_)
        disp2 = self.dec_disp2(h2_)
        pi2 = self.dec_pi2(h2_)
        
        h10 = self.encoder1(x1)
        h20 = self.encoder2(x2)
        
        combine_latent0 = torch.cat([h10, h20], dim=-1)
        z0 = self.latent_enc(combine_latent0)
        combine_latent0_ = self.latent_dec(z0)
        q = self.soft_assign(z0)
        num, lq = self.cal_latent(z0)
        return z0, q, num, lq, mean1, mean2, disp1, disp2, pi1, pi2, combine_latent0, combine_latent0_, h10, h20
        
    def forward_AE(self, x1, x2):
        h1 = self.encoder1(x1+torch.randn_like(x1) * self.sigma1)
        h2 = self.encoder2(x2+torch.randn_like(x2) * self.sigma2)
        
        h1_ = self.decoder1(h1)
        mean1 = self.dec_mean1(h1_)
        disp1 = self.dec_disp1(h1_)
        pi1 = self.dec_pi1(h1_)
        
        h2_ = self.decoder2(h2)
        mean2 = self.dec_mean2(h2_)
        disp2 = self.dec_disp2(h2_)
        pi2 = self.dec_pi2(h2_)       
        
        h10 = self.encoder1(x1)
        h20 = self.encoder2(x2)
        
        combine_latent0 = torch.cat([h10, h20], dim=-1)
        z0 = self.latent_enc(combine_latent0)
        combine_latent0_ = self.latent_dec(z0)
        num, lq = self.cal_latent(z0)
        return z0, num, lq, mean1, mean2, disp1, disp2, pi1, pi2, combine_latent0, combine_latent0_, h10, h20
        
    def encodeBatch(self, X1, X2, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            
        encoded = []
        h1_ = []
        h2_ = []
        self.eval()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs1 = Variable(x1batch)
            inputs2 = Variable(x2batch)
            z,_,_,_,_,_,_,_,_,_,_,h10,h20 = self.forward_AE(inputs1, inputs2)
            encoded.append(z.data)
            h1_.append(h10.data)
            h2_.append(h20.data)

        encoded = torch.cat(encoded, dim=0)
        h1_ = torch.cat(h1_, dim=0)
        h2_ = torch.cat(h2_, dim=0)
        return encoded, h1_, h2_

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss
    
    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q))
        c2 = -torch.sum(p * torch.log(p))
        l = c1 - c2
        return l

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return cl_loss

    def pretrain_autoencoder(self, X1, X_raw1, sf1, X2, X_raw2, sf2, 
            batch_size=256, lr=0.001, epochs=5, ae_save=True, ae_weights='AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X_raw1), torch.Tensor(sf1), torch.Tensor(X2), torch.Tensor(X_raw2), torch.Tensor(sf2))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                x1_tensor = Variable(x1_batch).cuda()
                x_raw1_tensor = Variable(x_raw1_batch).cuda()
                sf1_tensor = Variable(sf1_batch).cuda()
                x2_tensor = Variable(x2_batch).cuda()
                x_raw2_tensor = Variable(x_raw2_batch).cuda()
                sf2_tensor = Variable(sf2_batch).cuda()
                zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor, combine_latent0, combine_latent0_,_,_ = self.forward_AE(x1_tensor, x2_tensor)
                recon_loss1 = self.zinb_loss(x=x_raw1_tensor, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_tensor)
                recon_loss2 = self.zinb_loss(x=x_raw2_tensor, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sf2_tensor)
                recon_loss_latent = self.mse(combine_latent0_, combine_latent0) 
                if epoch >= epochs * self.cutoff1 and batch_idx >= num_batch * self.cutoff2:
                   loss = recon_loss1 + recon_loss2 + recon_loss_latent * self.beta
                else:
                   loss = recon_loss1 + recon_loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}, NB loss:{:.4f}, latent MSE loss:{:.8f}'.format(
                batch_idx+1, epoch+1, recon_loss1.item(), recon_loss2.item(), recon_loss_latent.item()))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X1, X_raw1, sf1, X2, X_raw2, sf2, ml_ind1=np.array([]), ml_ind2=np.array([]), cl_ind1=np.array([]), cl_ind2=np.array([]), ml_p=1., cl_p=1., y=None, lr=0.001,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, ae_save=True, hcae_weights="HCAE_weights.pth.tar", save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        n_clusters = self.n_clusters
        X1 = torch.tensor(X1).cuda()
        X_raw1 = torch.tensor(X_raw1).cuda()
        sf1 = torch.tensor(sf1).cuda()
        X2 = torch.tensor(X2).cuda()
        X_raw2 = torch.tensor(X_raw2).cuda()
        sf2 = torch.tensor(sf2).cuda()
       # self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim))
       # optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
             
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters, n_init=30)
        Zdata,_,_ = self.encodeBatch(X1, X2, batch_size=batch_size)
        #latent
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
           # self.y_pred_ = best_map(y, self.y_pred)
           # acc = np.round(metrics.accuracy_score(y, self.y_pred_), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5) 
            print('Initializing k-means: NMI= %.4f, ARI= %.4f, AMI= %.4f' % (nmi, ari, ami))
        
        self.train()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        ml_num_batch = int(math.ceil(1.0 * ml_ind1.shape[0] / batch_size))
        ml_num = ml_ind1.shape[0]
        cl_num_batch = int(math.ceil(1.0 * cl_ind1.shape[0] / batch_size))
        cl_num = cl_ind1.shape[0]

        final_nmi, final_ari, final_ami, final_epoch = 0, 0, 0, 0
        update_ml = 1
        update_cl = 1

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                Zdata,_,_ = self.encodeBatch(X1, X2, batch_size=batch_size)
                q = self.soft_assign(Zdata)
                p = self.target_distribution(q).data
                
                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                
                if y is not None:
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    final_ami = ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                    print('Clustering   %d: NMI= %.4f, ARI= %.4f, AMI= %.4f' % (epoch+1, nmi, ari,ami))

                # save current model
                # if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    # self.save_checkpoint({'epoch': epoch+1,
                            # 'state_dict': self.state_dict(),
                            # 'mu': self.mu,
                            # 'p': p,
                            # 'q': q,
                            # 'y_pred': self.y_pred,
                            # 'y_pred_last': self.y_pred_last,
                            # 'y': y
                            # }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                
                if delta_label < tol or epoch == num_epochs-1:
                    if ae_save:
                       torch.save({'ae_state_dict': self.state_dict(),
                                   'optimizer_state_dict': optimizer.state_dict()}, hcae_weights)
                
                if epoch>20 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break
                
            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss1_val = 0.0
            recon_loss2_val = 0.0
            recon_loss_latent_val = 0.0
            cluster_loss_val = 0.0
            kl_loss_val = 0.0
            for batch_idx in range(num_batch):
                x1_batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw1_batch = X_raw1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf1_batch = sf1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x2_batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw2_batch = X_raw2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf2_batch = sf2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs1 = Variable(x1_batch)
                rawinputs1 = Variable(x_raw1_batch)
                sfinputs1 = Variable(sf1_batch)
                inputs2 = Variable(x2_batch)
                rawinputs2 = Variable(x_raw2_batch)
                sfinputs2 = Variable(sf2_batch)
                target1 = Variable(pbatch)

                zbatch, qbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor, combine_latent0, combine_latent0_, _, _ = self.forward(inputs1, inputs2)               
                cluster_loss = self.cluster_loss(target1, qbatch)
                recon_loss1 = self.zinb_loss(x=rawinputs1, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sfinputs1)
                recon_loss2 = self.zinb_loss(x=rawinputs2, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sfinputs2)
                recon_loss_latent = self.mse(combine_latent0_, combine_latent0)
                loss = recon_loss1 + recon_loss2 + recon_loss_latent * self.beta + cluster_loss * self.gamma
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mu, 1)
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs1)
                recon_loss1_val += recon_loss1.data * len(inputs1)
                recon_loss2_val += recon_loss2.data * len(inputs2)
                recon_loss_latent_val += recon_loss_latent.data * len(inputs1)
                train_loss = recon_loss1_val + recon_loss2_val + recon_loss_latent_val + cluster_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.8f ZINB Loss1: %.4f ZINB Loss2: %.4f Latent MSE Loss: %.4f" % (
               epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss1_val / num, recon_loss2_val / num, recon_loss_latent_val / num))
               
            ml_loss = 0.0
            if epoch % update_ml == 0 and epoch>0:
                for ml_batch_idx in range(ml_num_batch):
                    px1_1 = X1[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pxraw1_1 = X_raw1[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    sf1_1 = sf1[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    px2_1 = X2[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw2_1 = X_raw2[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf2_1 = sf2[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]

                    px1_2 = X1[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    sf1_2 = sf1[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pxraw1_2 = X_raw1[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    px2_2 = X2[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf2_2 = sf2[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw2_2 = X_raw2[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]

                    optimizer.zero_grad()
                    inputs1_1 = Variable(px1_1)
                    rawinputs1_1 = Variable(pxraw1_1)
                    sfinputs1_1 = Variable(sf1_1)
                    inputs1_2 = Variable(px1_2)
                    rawinputs1_2 = Variable(pxraw1_2)
                    sfinputs1_2 = Variable(sf1_2)

                    inputs2_1 = Variable(px2_1)
                    rawinputs2_1 = Variable(pxraw2_1)
                    sfinputs2_1 = Variable(sf2_1)
                    inputs2_2 = Variable(px2_2)
                    rawinputs2_2 = Variable(pxraw2_2)
                    sfinputs2_2 = Variable(sf2_2)

                    zbatch1, qbatch1, z_num1, lqbatch1, mean1_tensor_1, mean2_tensor_1, disp1_tensor_1, disp2_tensor_1, pi1_tensor_1, pi2_tensor_1, combine_latent1, combine_latent1_, _, _ = self.forward(inputs1_1,inputs2_1)
                    zbatch2, qbatch2, z_num2, lqbatch2, mean1_tensor_2, mean2_tensor_2, disp1_tensor_2, disp2_tensor_2, pi1_tensor_2, pi2_tensor_2, combine_latent2, combine_latent2_, _, _ = self.forward(inputs1_2,inputs2_2)

                    recon_loss1_1 = self.zinb_loss(x=rawinputs1_1, mean=mean1_tensor_1, disp=disp1_tensor_1, pi=pi1_tensor_1,scale_factor=sfinputs1_1)
                    recon_loss1_2 = self.zinb_loss(x=rawinputs1_2, mean=mean1_tensor_2, disp=disp1_tensor_2, pi=pi1_tensor_2,scale_factor=sfinputs1_2)
                    recon_loss_latent_1 = self.mse(combine_latent1, combine_latent1_)

                    recon_loss2_1 = self.zinb_loss(x=rawinputs2_1, mean=mean2_tensor_1, disp=disp2_tensor_1, pi=pi2_tensor_1,scale_factor=sfinputs2_1)
                    recon_loss2_2 = self.zinb_loss(x=rawinputs2_2, mean=mean2_tensor_2, disp=disp2_tensor_2, pi=pi2_tensor_2,scale_factor=sfinputs2_2)
                    recon_loss_latent_2 = self.mse(combine_latent2, combine_latent2_)
                    
                    loss_ = ml_p*self.pairwise_loss(qbatch1, qbatch2, "ML")

                    loss = loss_ + recon_loss1_1 + recon_loss1_2 +recon_loss2_1 + recon_loss2_2+ recon_loss_latent_1 * self.beta + recon_loss_latent_2 * self.beta
                    ml_loss += loss_.data
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.mu, 1)
                    optimizer.step()

            cl_loss = 0.0
            if epoch % update_cl == 0 and epoch>0:
                for cl_batch_idx in range(cl_num_batch):
                    px1_1 = X1[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    px2_1 = X2[cl_ind1[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]

                    px1_2 = X1[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    px2_2 = X2[cl_ind2[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]

                    optimizer.zero_grad()
                    inputs1_1 = Variable(px1_1)
                    inputs1_2 = Variable(px1_2)
                    inputs2_1 = Variable(px2_1)
                    inputs2_2 = Variable(px2_2)

                    _, qbatch1, _, _, _, _, _, _, _, _, _, _, _, _ = self.forward(inputs1_1,inputs2_1)
                    _, qbatch2, _, _, _, _, _, _, _, _, _, _, _, _ = self.forward(inputs1_2,inputs2_2)

                    loss = cl_p*self.pairwise_loss(qbatch1, qbatch2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.mu, 1)
                    optimizer.step()
                    
            if ml_num_batch >0 and cl_num_batch >0 and epoch>0:
                print("ML loss %.4f CL loss %.4f" % (float(ml_loss.cpu()),float(cl_loss.cpu())))

        return self.y_pred, final_nmi, final_ari, final_ami, final_epoch
