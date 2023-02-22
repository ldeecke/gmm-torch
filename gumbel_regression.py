import numpy as np
import torch
from scipy.special import logsumexp


def gumbel_loss_numpy(diff, beta):
    '''
    diff: y - x, where x is prediction and y is the label
    beta: temperature (scale parameter) for the gumbel
    '''
    return np.exp(diff/beta) - diff/beta - 1

def gumbel_loss(diff, beta):
    z = diff/beta
    loss = torch.exp(z) - z - 1
    return loss

def gumbel_stable_loss(diff, beta, clip=None):
    z = diff/beta
    if clip is not None:
        z = torch.clamp(z, max=clip)

    max_z = torch.max(z)
    max_z = torch.where(max_z < -1.0, torch.tensor(-1.0, dtype=torch.double, device=max_z.device), max_z)
    max_z = max_z.detach()  # Detach the gradients

    # NOTE(@motiwari): There might be a sign error below
    loss = torch.exp(z - max_z) - z*torch.exp(-max_z) - torch.exp(-max_z)    # scale by e^max_z
    return loss

def log_partition(x, beta):
    """Analytically calculate the Log-Partition over an empirical distribution
    """
    n = x.shape[0]
    return beta * logsumexp(x / beta) - beta * np.log(n)


def solver_1d(data, loss_fn, lr=0.005, batch=128, steps=4000):
    """ Estimate Log-Partition function using Gumbel loss with SGD
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    statistic = torch.zeros(1, requires_grad=True)
    optim = torch.optim.SGD([statistic], lr=lr)

    for _ in range(steps):
        # Sample a batch
        perm = torch.randperm(data.size(0))
        idx = perm[:batch]
        samples = data[idx]

        optim.zero_grad()
        loss = torch.mean(loss_fn(samples - statistic))
        loss.backward()
        optim.step()

    return statistic.detach().cpu().numpy()



