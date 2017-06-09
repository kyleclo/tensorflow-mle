# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# visualize results
import numpy as np
import scipy.stats as sp
from util.dist import get_mu, get_sigma
import matplotlib.pyplot as plt


def plot_canonical_gauss(x, obs_mu, obs_sigma, obs_loss,
                         title, epsilon=0.05, breaks=100):
    # compute grid
    mu_grid = np.linspace(start=min(obs_mu) - epsilon,
                          stop=max(obs_mu) + epsilon,
                          num=breaks)
    sigma_grid = np.linspace(start=max(min(obs_sigma) - epsilon, 0.0),
                             stop=max(obs_sigma) + epsilon,
                             num=breaks)
    mu_grid, sigma_grid = np.meshgrid(mu_grid, sigma_grid)
    loss_grid = -np.sum(
        [sp.norm(loc=mu_grid, scale=sigma_grid).logpdf(x=xi) for xi in x],
        axis=0)

    # plot contours and loss
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].contour(mu_grid, sigma_grid, loss_grid,
                  levels=np.linspace(np.min(loss_grid),
                                     np.max(loss_grid),
                                     breaks),
                  cmap='terrain')
    ax[0].plot(obs_mu, obs_sigma, color='red', alpha=0.5,
               linestyle='dashed', linewidth=1, marker='.', markersize=3)
    ax[0].set_xlabel('mu')
    ax[0].set_ylabel('sigma')
    ax[1].plot(range(len(obs_loss)), obs_loss)
    ax[1].set_xlabel('iter')
    # ax[1].set_ylabel('loss')
    plt.suptitle('{}'.format(title))
    plt.show()


def plot_natural_gauss(x, obs_eta1, obs_eta2, obs_loss,
                       title, epsilon=0.05, breaks=300):
    # compute grid
    eta1_grid = np.linspace(start=min(obs_eta1) - epsilon,
                            stop=max(obs_eta1) + epsilon,
                            num=breaks)
    eta2_grid = np.linspace(start=min(obs_eta2) - epsilon,
                            stop=min(max(obs_eta2) + epsilon, 0.0),
                            num=breaks)

    eta1_grid, eta2_grid = np.meshgrid(eta1_grid, eta2_grid)

    mu_grid = get_mu(eta1_grid, eta2_grid)
    sigma_grid = get_sigma(eta2_grid)

    loss_grid = -np.sum(
        [sp.norm(loc=mu_grid, scale=sigma_grid).logpdf(x=xi) for xi in x],
        axis=0)

    # plot contours and loss
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].contour(eta1_grid, eta2_grid, loss_grid,
                  levels=np.linspace(np.min(loss_grid),
                                     np.max(loss_grid),
                                     breaks),
                  cmap='terrain')
    ax[0].plot(obs_eta1, obs_eta2, color='red', alpha=0.5,
               linestyle='dashed', linewidth=1, marker='.', markersize=3)
    ax[0].set_xlabel('eta1')
    ax[0].set_ylabel('eta2')
    ax[1].plot(range(len(obs_loss)), obs_loss)
    ax[1].set_xlabel('iter')
    # ax[1].set_ylabel('loss')
    plt.suptitle('{}'.format(title))
    plt.show()


