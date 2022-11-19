from pathlib import Path

import matplotlib
#matplotlib.use('Agg')

matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'legend.framealpha': 0.5,
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}'
})
import matplotlib.pyplot as plt
import numpy as np

from .. import utils
from .. import custom_paths


def get_data(n_train, n_pool):
    n = n_train + n_pool
    x = []
    means = [[0, 0], [2, 0], [4, 0]]
    vars = [[0.5, 1], [2, 0.1], [0.5, 1]]
    probs = [0.7, 0.2, 0.1]
    for i in range(n):
        idx = np.random.choice([0, 1, 2], p=probs)
        x.append(np.asarray(means[idx]) + np.asarray(vars[idx]) * np.random.randn(2))
    x = np.asarray(x)
    return x[:n_train], x[n_train:]


def plot_step(x_train, x_pool, filename: str):
    fix, ax = plt.subplots(figsize=(3.5, 2.6))
    plt.tight_layout()
    sq_dists = np.linalg.norm(x_train[:, None, :] - x_pool[None, :, :], axis=-1)**2
    cluster_idxs = np.argmin(sq_dists, axis=0)
    min_sq_dists = np.min(sq_dists, axis=0)
    cluster_sizes = np.bincount(cluster_idxs, weights=min_sq_dists, minlength=x_train.shape[0])
    max_cluster_size = np.max(cluster_sizes)
    is_in_largest_cluster = cluster_sizes[cluster_idxs] == max_cluster_size
    min_sq_dists[~is_in_largest_cluster] = -np.Inf
    best_idx = np.argmax(min_sq_dists)
    for i in range(x_pool.shape[0]):
        color = '#FF8800' if is_in_largest_cluster[i] else '#4444FF'
        point = x_pool[i, :]
        center = x_train[cluster_idxs[i], :]
        plt.plot([center[0], point[0]], [center[1], point[1]], '-', color=color, alpha=0.2)
        # if i == best_idx:
        #     plt.plot([point[0]], [point[1]], '.', color='r')
    plt.plot(x_pool[:, 0], x_pool[:, 1], '.', color='#AAAAAA', alpha=0.6, markeredgewidth=0)
    plt.plot(x_pool[best_idx, 0], x_pool[best_idx, 1], '.', color='r', markeredgewidth=0)
    plt.plot(x_train[:, 0], x_train[:, 1], '.', color='k', markeredgewidth=0)
    ax.set_aspect('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return best_idx


def create_lcmd_plots(n_train, n_pool, n_steps=2, use_png=False):
    np.random.seed(0)
    x_train, x_pool = get_data(n_train, n_pool)
    for i in range(n_steps):
        plots_path = Path(custom_paths.get_plots_path())
        if use_png:
            filename = plots_path / 'lcmd_visualization_png' / f'lcmd_{n_train}_{n_pool}_{i}.png'
        else:
            filename = plots_path / 'lcmd_visualization' / f'lcmd_{n_train}_{n_pool}_{i}.pdf'
        utils.ensureDir(filename)
        best_idx = plot_step(x_train, x_pool, filename=str(filename))
        x_train = np.concatenate([x_train, x_pool[best_idx:best_idx+1, :]], axis=0)
        x_pool = np.delete(x_pool, best_idx, axis=0)

