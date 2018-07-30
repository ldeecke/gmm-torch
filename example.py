import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", font="Arial")
colors = sns.color_palette("Paired", n_colors=12).as_hex()

import numpy as np
import torch

from gmm import GaussianMixture
from math import sqrt


def main():
    n, d = 400, 2

    # generate some data points ..
    data = torch.Tensor(n, d).normal_()
    # .. as well as a random partition that ..
    ids = np.random.choice(n, n//2, replace=False)
    # .. is permuted to come from a non-standard Gaussian N(7, 16)
    data[:n//2] -= 1
    data[:n//2] *= sqrt(3)
    data[n//2:] += 1
    data[n//2:] *= sqrt(2)

    # a Gaussian Mixture Model is instantiated and ..
    n_components = 2
    model = GaussianMixture(n_components, d)
    model.fit(data)
    # .. used to predict the data points that where shifted
    y = model.predict(data)
    c = np.isin(np.where(y > 0), ids)

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    ax.set_facecolor('#bbbbbb')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    # plot the locations of all data points ..
    ax.scatter(*data[:n//2].data.t(), color="#000000", s=3, alpha=.75, label="Ground-truth 1")
    ax.scatter(*data[n//2:].data.t(), color="#ffffff", s=3, alpha=.75, label="Ground-truth 2")

    # .. and circle them according to their classification
    ax.scatter(*data[np.where(y == 0)].data.t(), zorder=0, color="#dbe9ff", alpha=.6, edgecolors=colors[1], label="Predicted 1")
    ax.scatter(*data[np.where(y == 1)].data.t(), zorder=0, color="#ffdbdb", alpha=.6, edgecolors=colors[5], label="Predicted 2")

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("example.pdf")

if __name__ == "__main__":
    main()
