import torch
import numpy as np

from math import pi
from torch import nn

class FisherVectorLayer(nn.Module):
    def __init__(self, gmm):
        super(FisherVectorLayer, self).__init__()
        self.gmm = gmm

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        Q = torch.zeros(B, N, self.gmm.n_components)

        for vid in range(B):
            Q[vid] = self.gmm.predict_proba(x[vid])
        Q_sum = torch.sum(Q, 1).unsqueeze(2) / N  # B, K, 1
        Q_x = torch.einsum('knb,bnd->bkd', Q.T, x) / N  # B, K, D
        Q_x_2 = torch.einsum('knb,bnd->bkd', Q.T, x ** 2) / N # B, K, D

        d_pi = Q_sum - self.gmm.pi  #find weights # B, K
        d_mu = Q_x - Q_sum * self.gmm.mu #B, K, D
        d_sigma = (- Q_x_2 - Q_sum * self.gmm.mu ** 2 + Q_sum * self.gmm.var + 2 * Q_x * self.gmm.mu) #B, K, D

        # Merge derivatives into a vector.
        return torch.hstack((d_pi.reshape(B, -1), d_mu.reshape(B, -1), d_sigma.reshape(B, -1)))
