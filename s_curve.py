from sklearn import datasets, manifold, mixture, metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats, linalg
import itertools
from math import ceil


def make_s_curve(n_samples, seed):
    x, y = datasets.make_s_curve(n_samples=n_samples, random_state=seed)
    idx = y.argsort()
    y.sort()
    x = x[idx]
    y = []
    for i in range(n_samples):
        if i < n_samples / 2:
            y.append('purple')
        else:
            y.append('blue')
    return (x, y)

def plot_s_curve(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_vals = x[:, 0]
    z_vals = x[:, 1]
    y_vals = x[:, 2]
    ax.scatter(x_vals, y_vals, z_vals, c=y)
    plt.show()

def plot_isomap(y, color, p):
    fig = plt.figure()
    gs = fig.add_gridspec(1 + len(y) // p, p)
    axs = []
    for i in range(len(y)):
        row = i // p
        col = i % p
        axs.append(fig.add_subplot(gs[row, col]))
        x_vals = y[i][:, 0]
        y_vals = y[i][:, 1]
        axs[i].scatter(x_vals, y_vals, c=color)
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
    plt.show()

def evaluate(y):
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    ari = []
    for i in range(len(y)):
        gmm.fit(y[i])
        ari.append(metrics.adjusted_rand_score(true_labels, gmm.predict(y[i])))
        print(i, ari[i])
    fig = plt.figure()
    ax = fig.add_subplot()
    x_vals = neighbors_range
    ax.scatter(x_vals, ari)
    plt.show()

if __name__ == "__main__":
    n_samples = 500
    p = 3
    custom_range = range(5, 50)
    seed = 1

    x, y = make_s_curve(n_samples, seed)
    # plot_s_curve(x, y)

    true_labels = []
    for c in y:
        if c == 'purple':
            true_labels.append(0)
        else:
            true_labels.append(1)
    true_labels.reverse()

    y_isomap = []
    neighbors_range = [i for i in custom_range if i % p == 0]
    for i in neighbors_range:
        isomap = manifold.Isomap(n_neighbors=i, n_components=2)
        y_isomap.append(isomap.fit_transform(x))
    plot_isomap(y_isomap, y, p)

    evaluate(y_isomap)
