"""Markov Chain Monte Carlo Plotting and Utilities

Copyright (c) Alex Gorodetsky, 2020
License: MIT
"""
import numpy as np
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import scipy
from scipy import stats

import matplotlib as mpl
from matplotlib import cm
from collections import OrderedDict
from collections import namedtuple
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# For comments on the first four functions see the Gaussian Random Variable Notebook
def lognormpdf(x, mean, cov):
    d, N = x.shape
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.linalg.det(cov)**0.5
    diff = x - np.tile(mean[:, np.newaxis], (1, N))
    sol = np.linalg.solve(cov, diff)
    inexp = np.einsum("ij,ij->j",diff, sol)
    out = np.log(preexp) - 0.5 * inexp
    return out

def build_cov_mat(std1, std2, rho):
    """Build a covariance matrix for a bivariate Gaussian distribution
    
    Inputs
    ------
    std1 : positive real, standard deviation of first variable
    std2 : positive real, standard deviation of second variable
    rho  : real number between [-1, 1] representing the correlation
    
    Returns
    -------
    Bivariate covariance Matrix
    """
    assert std1 > 0, "standard deviation must be greater than 0"
    assert std2 > 0, "standard deviation must be greater than 0"
    assert np.abs(rho) <= 1, "correlation must be betwene -1 and 1"
    return np.array([[std1**2, rho * std1 * std2], [rho * std1 * std2, std2**2]])

def eval_normpdf_on_grid(x, y, mean, cov):
    XX, YY = np.meshgrid(x,y)
    pts = np.stack((XX.reshape(-1), YY.reshape(-1)),axis=0)
    evals = np.exp(lognormpdf(pts, mean, cov).reshape(XX.shape))
    return XX, YY, evals

def eval_func_on_grid(func, gridx, gridy):
    "Evaluate the function *func* on a grid discretized by gridx and gridy"
    vals = np.zeros((gridx.shape[0], gridy.shape[0]))
    for ii in range(gridx.shape[0]):
        for jj in range(gridy.shape[0]):
            pt = np.array([gridx[ii], gridy[jj]])
            vals[ii, jj] = func(pt)
            
    return vals   

def plot_bivariate_gauss(x, y, mean, cov, axis=None):
    std1 = cov[0,0]**0.5
    std2 = cov[1,1]**0.5
    mean1 = mean[0]
    mean2 = mean[1]
    XX, YY, evals = eval_normpdf_on_grid(x, y, mean, cov)
    if axis is None:
        fig, axis = plt.subplots(2,2, figsize=(10,10))
        
    axis[0,0].plot(x, np.exp(lognormpdf(x[np.newaxis,:], np.array([mean1]), np.array([[std1**2]]))))
    axis[0,0].set_ylabel(r'$f_{X_1}$')
    axis[1,1].plot(np.exp(lognormpdf(y[np.newaxis,:], np.array([mean2]), np.array([[std2**2]]))),y)
    axis[1,1].set_xlabel(r'$f_{X_2}$')
    axis[1,0].contourf(XX, YY, evals)
    axis[1,0].set_xlabel(r'$x_1$')
    axis[1,0].set_ylabel(r'$x_2$')
    axis[0,1].set_visible(False)
    return fig, axis

def sub_sample_data(samples, frac_burn=0.2, frac_use=0.7):
    """Subsample data by burning off the front fraction and using another fraction

    Inputs
    ------
    samples: (N, d) array of samples
    frac_burn: fraction < 1, percentage of samples from the front to ignore
    frac_use: percentage of samples to use after burning, uniformly spaced
    """
    nsamples = samples.shape[0]
    inds = np.arange(nsamples, dtype=np.int)
    start = int(frac_burn * nsamples)
    inds = inds[start:]
    nsamples = nsamples - start
    step = int(nsamples / (nsamples * frac_use))
    inds2 = np.arange(0, nsamples, step)
    inds = inds[inds2]
    return samples[inds, :]

def scatter_matrix(samples, #list of chains
                   mins=None, maxs=None,
                   upper_right=None,
                   specials=None,
                   hist_plot=True, # if false then only data
                   nbins=200,
                   gamma=0.5,
                   labels=None):

    nchains = len(samples)
    dim = samples[0].shape[1]

    if mins is None:
        mins = np.zeros((dim))
        maxs = np.zeros((dim))

        for ii in range(dim):
            # print("ii = ", ii)
            mm = [np.quantile(samp[:, ii], 0.01, axis=0) for samp in samples]
            # print("\t mins = ", mm)
            mins[ii] = np.min(mm)
            mm = [np.quantile(samp[:, ii], 0.99, axis=0) for samp in samples]            
            # print("\t maxs = ", mm)
            maxs[ii] = np.max(mm)

            if specials is not None:
                if isinstance(specials, list):
                    minspec = np.min([spec['vals'][ii] for spec in specials])
                    maxspec = np.max([spec['vals'][ii] for spec in specials])
                else:
                    minspec = spec['vals'][ii]
                    maxspec = spec['vals'][ii]
                mins[ii] = min(mins[ii], minspec)
                maxs[ii] = max(maxs[ii], maxspec)
    

    deltas = (maxs - mins) / 10.0
    use_mins = mins - deltas
    use_maxs = maxs + deltas

    cmuse = cm.get_cmap(name='tab10')

    # fig = plt.figure(constrained_layout=True)
    fig = plt.figure()
    if upper_right is None:
        gs = GridSpec(dim, dim, figure=fig)
        axs = [None]*dim*dim
        start = 0
        end = dim
        l = dim
    else:
        gs = GridSpec(dim+1, dim+1, figure=fig)
        axs = [None]*(dim+1)*(dim+1)
        start = 1
        end = dim + 1
        l = dim+1

    # print("mins = ", mins)
    # print("maxs = ", maxs)

    for ii in range(dim):
        # print("ii = ", ii)
        axs[ii] = fig.add_subplot(gs[ii+start, ii])
        ax = axs[ii]

        # Turn everythinng off
        if ii < dim-1:
            ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
            if labels:
                ax.set_xlabel(labels[ii])
            
        ax.tick_params(axis='y', left=False, right=False, labelleft=False)
        ax.set_frame_on(False)

        sampii = np.concatenate([samples[kk][:, ii] for kk in range(nchains)])
        # for kk in range(nchains):
        # print("sampii == ", sampii)
        ax.hist(sampii,            
                # ax.hist(samples[kk][:, ii],
                bins='sturges',
                density=True,
                edgecolor='black',
                stacked=True,
                range=(use_mins[ii],use_maxs[ii]),
                alpha=0.4)
        if specials is not None:
            for special in specials:
                if special['vals'][ii] is not None:
                    # ax.axvline(special[ii], color='red', lw=2)
                    if 'color' in special:
                        ax.axvline(special['vals'][ii], color=special['color'], lw=2)
                    else:
                        ax.axvline(special['vals'][ii], lw=2)

        ax.set_xlim((use_mins[ii]-1e-10, use_maxs[ii]+1e-10))

        for jj in range(ii+1, dim):
            # print("jj = ", jj)
            axs[jj*l + ii] = fig.add_subplot(gs[jj+start, ii])
            ax = axs[jj*l + ii]


            if jj < dim-1:
                ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
                if labels:
                    ax.set_xlabel(labels[ii])
            if ii > 0:
                ax.tick_params(axis='y', left=False, right=False, labelleft=False)
            else:
                ax.tick_params(axis='y', left=True, right=False, labelleft=True)
                if labels:
                    ax.set_ylabel(labels[jj])
                    
            ax.set_frame_on(True)     

            for kk in range(nchains):
                if hist_plot is True:
                    ax.hist2d(samples[kk][:, ii], samples[kk][:, jj],
                              bins=nbins,
                              norm=mcolors.PowerNorm(gamma),
                              density=True)
                else:
                    ax.plot(samples[kk][:, ii], samples[kk][:, jj], 'o', ms=1, alpha=gamma)

                # ax.hist2d(samples[kk][:, ii], samples[kk][:, jj], bins=nbins)

            if specials is not None:
                for special in specials:
                    if 'color' in special:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x',
                                color=special['color'], ms=2, mew=2)
                    else:
                        ax.plot(special['vals'][ii], special['vals'][jj], 'x',
                                ms=2, mew=2)


            ax.set_xlim((use_mins[ii], use_maxs[ii]))
            ax.set_ylim((use_mins[jj]-1e-10, use_maxs[jj]+1e-10))



    plt.tight_layout(pad=0.01);
    if upper_right is not None:
        size_ur = int(dim/2)

        name = upper_right['name']
        vals = upper_right['vals']
        if 'log_transform' in upper_right:
            log_transform = upper_right['log_transform']
        else:
            log_transform = None
        ax = fig.add_subplot(gs[0:int(dim/2),
                                size_ur+1:size_ur+int(dim/2)+1])

        lb = np.min([np.quantile(val, 0.01) for val in vals])
        ub = np.max([np.quantile(val, 0.99) for val in vals])
        for kk in range(nchains):
            if log_transform is not None:
                pv = np.log10(vals[kk]) 
                ra = (np.log10(lb), np.log10(ub))
            else:
                pv = vals[kk]
                ra = (lb, ub)
            ax.hist(pv,
                    density=True,
                    range=ra,
                    edgecolor='black',
                    stacked=True,
                    bins='auto',
                    alpha=0.2)
        ax.tick_params(axis='x', bottom='both', top=False, labelbottom=True)
        ax.tick_params(axis='y', left='both', right=False, labelleft=False)
        ax.set_frame_on(True)
        ax.set_xlabel(name)
    plt.subplots_adjust(left=0.15, right=0.95)
    return fig, axs, gs
