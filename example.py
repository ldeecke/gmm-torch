import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", font="Arial")

import torch

from gmm import GaussianMixture
from math import sqrt


def main():
    n, d = 300, 2

    # generate some data points ..
    data = torch.Tensor(n, d).normal_()
    # .. and shift them around to non-standard Gaussians
    data[:n//2] -= 1
    data[:n//2] *= sqrt(3)
    data[n//2:] += 1
    data[n//2:] *= sqrt(2)

    # Next, the Gaussian mixture is instantiated and ..
    n_components = 2
    model = GaussianMixture(n_components, d)
    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)

    plot(data, y)


def plot(data, true_y, pred_y, iter, mus):
    n = true_y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    colors = ['black', 'white', 'red']

    for point_idx, point in enumerate(data.data):
        true_label = true_y[point_idx]
        pred_label = pred_y[point_idx]
        ax.scatter(*point, color=colors[true_label], s=3, alpha=0.75, zorder=n+point_idx)
        ax.scatter(*point, color=colors[pred_label], s=50, edgecolors=colors[pred_label], alpha=0.6, zorder=point_idx)

    for mu_idx, mu in enumerate(mus.data[0]):
        print(mus)
        print(mu)
        ax.scatter(*mu, color=colors[mu_idx], marker='x', s=100, zorder=2*n)

    # handles = [
    #     plt.Line2D([0], [0], color="white", lw=4, label="Ground Truth 1"),
    #     plt.Line2D([0], [0], color="black", lw=4, label="Ground Truth 2"),
    #     plt.Line2D([0], [0], color=colors[1], lw=4, label="Predicted 1"),
    #     plt.Line2D([0], [0], color=colors[5], lw=4, label="Predicted 2"),
    # ]
    #
    # legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    plt.savefig("example" + str(iter) + ".pdf")
    plt.close()


if __name__ == "__main__":
    main()
