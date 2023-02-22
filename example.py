import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", font="Arial")

import torch
import os

from gmm_exact import GaussianMixtureExact
from gmm_gumbel import GaussianMixtureGumbel
from math import sqrt

import numpy as np
import itertools

colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'purple']

def plot(data, true_y, pred_y, iter, mus, K):
    n = true_y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")


    for point_idx, point in enumerate(data.data):
        true_label = true_y[point_idx]
        pred_label = pred_y[point_idx]
        ax.scatter(*point, color=colors[true_label], s=3, alpha=0.75, zorder=n+point_idx)
        ax.scatter(*point, color=colors[pred_label], s=50, edgecolors=colors[pred_label], alpha=0.6, zorder=point_idx)

    # import ipdb; ipdb.set_trace()
    for mu_idx, mu in enumerate(mus):
        ax.scatter(*mu, color=colors[mu_idx], marker='x', s=100, zorder=2*n)

    handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label="Ground Truth " + str(i)) for i in range(K)]
    handles += [plt.Line2D([0], [0], color=colors[i], lw=4, label="Predicted " + str(i)) for i in range(K)]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    plt.savefig(os.path.join("examples", "example" + str(iter) + ".pdf"))
    plt.close()

def create_data_1(N, K, D):
    # generate some data points ..
    data = torch.Tensor(N, D).normal_()
    # .. and shift them around to non-standard Gaussians
    chunk_size = N // K

    # Note that this loops truncates the last chunk_size - 1 if K doesn't divide N
    true_mus = []
    true_ys = []
    for cluster in range(K):
        true_ys += [cluster] * chunk_size

        # Shift each coordinate by -cluster
        data[cluster * chunk_size:(cluster + 1) * chunk_size] -= 2 * cluster

        # Even cluster indices have sqrt(2), odd have sqrt(3)

        if cluster % 2 == 0:
            sigma = 2
        else:
            sigma = 3

        data[cluster * chunk_size:(cluster + 1) * chunk_size] *= sqrt(sigma)
        new_mu = np.array([-2 * sqrt(sigma) * cluster] * D)
        true_mus.append(new_mu)

    return data, true_ys, true_mus

def find_best_permutation(true_ys, pred_ys, K):
    perms = list(itertools.permutations(range(K)))
    best_acc = 0
    best_perm = None
    best_pred_ys = None

    # Do reassignment
    for perm in perms:
        new_pred_ys = -np.ones(len(true_ys))  # Can't do in-place
        for i in range(K):
            new_pred_ys[np.where(pred_ys == i)] = perm[i]
        new_acc = np.mean(new_pred_ys == true_ys)
        if new_acc > best_acc:
            best_acc = new_acc
            best_perm = perm
            best_pred_ys = new_pred_ys.copy()

    return best_pred_ys.astype('int32')


def main():
    N = 300
    K = 4
    D = 2
    data, true_ys, true_mus = create_data_1(N, K, D)

    # Next, the Gaussian mixture is instantiated and ..
    model = GaussianMixtureExact(K, D)
    model.fit(data)

    # .. used to predict the data points as they where shifted
    pred_ys = model.predict(data)
    # pred_ys = torch.zeros(N)
    true_ys = np.array(true_ys)
    true_mus = np.array(true_mus)

    best_pred_ys = find_best_permutation(true_ys, pred_ys, K)
    plot(data, true_ys, best_pred_ys, 0, true_mus, K)



if __name__ == "__main__":
    main()
