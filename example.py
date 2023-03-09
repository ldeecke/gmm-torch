import torch
import os

from gmm_exact import GaussianMixtureExact
from gmm_gumbel import GaussianMixtureGumbel
from math import sqrt

import numpy as np
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white", font="Arial")
colors = ['green', 'white', 'red', 'blue', 'yellow', 'purple']


def plot(data, true_y, pred_y, iter_, mus, K, model_str):
    n = true_y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    for point_idx, point in enumerate(data.data):
        true_label = true_y[point_idx]
        pred_label = pred_y[point_idx]
        ax.scatter(*point, color=colors[pred_label], s=3, alpha=0.75, zorder=n+point_idx)
        ax.scatter(*point, color=colors[true_label], s=50, edgecolors=colors[true_label], alpha=0.6, zorder=point_idx)

    # import ipdb; ipdb.set_trace()
    for mu_idx, mu in enumerate(mus):
        ax.scatter(*mu, color=colors[mu_idx], marker='X', edgecolors='black', s=100, zorder=2*n)

    handles = [plt.scatter([0], [0], color=colors[i], s=3, label="Predicted " + str(i)) for i in range(K)]
    handles += [plt.scatter([0], [0], color=colors[i], s=50, label="Ground Truth " + str(i)) for i in range(K)]

    legend = ax.legend(loc="best", handles=handles)
    plt.title("Accuracy: " + str(np.mean(true_y == pred_y)))

    plt.tight_layout()
    plt.savefig(os.path.join("examples", model_str, model_str + "example" + str(iter_) + ".pdf"))
    plt.close()


def create_data_1(N, K, D):
    # generate some data points
    data = torch.Tensor(N, D).normal_()
    #  shift them around to non-standard Gaussians
    chunk_size = N // K

    # Note that this loops truncates the last chunk_size - 1 if K doesn't divide N
    true_mus = []
    true_ys = []
    mu_multiplier = 2
    for cluster in range(K):
        true_ys += [cluster] * chunk_size

        # Shift each coordinate by -cluster
        data[cluster * chunk_size:(cluster + 1) * chunk_size] -= mu_multiplier * cluster

        # Even cluster indices have sqrt(2), odd have sqrt(3) stddev
        if cluster % 2 == 0:
            sigma = 2
        else:
            sigma = 3

        data[cluster * chunk_size:(cluster + 1) * chunk_size] *= sqrt(sigma)
        new_mu = np.array([-mu_multiplier * sqrt(sigma) * cluster] * D)
        true_mus.append(new_mu)

    return data, true_ys, true_mus


def find_best_permutation(true_ys, pred_ys, K):
    perms = list(itertools.permutations(range(K)))
    best_acc = 0
    best_pred_ys = None
    best_perm = None
    for perm in perms:
        new_pred_ys = -np.ones(len(true_ys))  # Can't do in-place

        # Reassign all labels
        for i in range(K):
            new_pred_ys[np.where(pred_ys == i)] = perm[i]

        new_acc = np.mean(new_pred_ys == true_ys)
        if new_acc > best_acc:
            best_acc = new_acc
            best_perm = perm
            best_pred_ys = new_pred_ys.copy()
    best_perm_inv = np.argsort(np.array(best_perm))
    return best_pred_ys.astype('int32'), best_perm, best_perm_inv


def main():
    torch.manual_seed(1337)
    np.random.seed(1337)

    N = 300
    K = 4
    D = 2
    data, true_ys, true_mus = create_data_1(N, K, D)

    for model_str in ["exact", "gumbel"]:
        if model_str == "exact":
            model = GaussianMixtureExact(K, D)
        else:
            model = GaussianMixtureGumbel(K, D)

        model.fit(data)
        pred_ys = model.predict(data)
        true_ys = np.array(true_ys)
        true_mus = np.array(true_mus)

        best_pred_ys, best_perm, best_perm_inv = find_best_permutation(true_ys, pred_ys, K)
        plot(data, true_ys, best_pred_ys, 0, true_mus, K, model_str)
        print(f"{model_str} mus:\n", model.mu[0][best_perm_inv])
        print(f"{model_str} score: ", model.score(data, as_average=True), "\n")
    print("True mus:", true_mus)


if __name__ == "__main__":
    main()
