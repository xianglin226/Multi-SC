import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math, os

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class MF(nn.Module):
    def __init__(self, input_dim, z_dim, w):
        super(MF, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, z_dim), nn.BatchNorm1d(z_dim, affine=False), MeanAct())
        self.encoder[0].weight = torch.nn.Parameter(torch.Tensor(w))
        self.mse = nn.MSELoss()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        return h

    def fit(self, X, B,
            batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(X),torch.Tensor(B))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            for batch_idx, (x_batch, b_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                b_tensor = b_batch.cuda()
                zbatch = self.forward(x_tensor)
                loss = self.mse(zbatch, b_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('MF epoch [{}/{}],  Loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)