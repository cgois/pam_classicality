import numpy as np
from numpy import sin, cos
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

from pam.classicality import par_preparations_classicality
import pam.utils


def xzpsi(params):
    """Preparation with vectors [x, z, (r, theta, phi)]."""

    # phi -> azimuth (horizontal on plot), theta -> polar
    phi, theta = params
    return np.array([[1, 0, 0],
                     [0, 0, 1],
                     [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]])



def plot(plot1_grid, plot1_vals, plot2_grid, plot2_vals):
    """Icosahedron and rombicuboc. heatmaps as subplots with a common color bar."""

    def cmap_plot(grid, vals, levels=10, vmin=None, vmax=None, edges=False,
                scatter=False, cmap=cm.Blues_r, ax=None):
        """Countour map plotting."""

        x, y = np.asarray(grid)
        z = np.reshape(vals, x.shape)
        if not ax:
            ax = plt

        im = ax.contourf(x, y, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        if edges:
            im = ax.contour(x, y, z, levels=levels, alpha=.7, linewidths=.4, colors='black')
        if scatter:
            im = ax.scatter(x, y, c=z, s=8, cmap=cmap)

        #sns.set(style='whitegrid')
        plt.ylabel(r"$\theta$ (rad)")
        plt.xlabel(r"$\phi$ (rad)")

        return im

    plt.figure(figsize=(24,7))
    sns.set(style='whitegrid')
    sns.set_context("paper", font_scale=2.5)
    matplotlib.rc('axes', edgecolor='black')

    vmin = np.min([plot1_vals, plot2_vals])
    vmax = np.max([plot1_vals, plot2_vals])
    lvls = np.linspace(vmin, vmax, 15)

    plt.subplot(121)
    cmap_plot(plot1_grid, plot1_vals, levels=lvls)

    plt.subplot(122)
    cmap_plot(plot2_grid, plot2_vals, levels=lvls)

    plt.subplots_adjust(bottom=.15, right=.8, top=.95)
    plt.colorbar(cax=plt.axes([.82, .15, .015, .8]), format="%.2f")
    plt.savefig(fname="heatmap.png")


def main(verb=-1):

    workers = 10
    icos = pam.utils.icos()
    romb = pam.utils.romb()
    ndetps_icos = 5000
    rounds_icos = 20
    ndetps_romb = int(.5e5)
    rounds_romb = 50

    nx, ny = 80, 40
    ma, mb = 2, 2
    parameter_grid = np.meshgrid(np.linspace(0, 2 * np.pi, nx),
                                 np.linspace(0, np.pi, ny))
    params_stack = np.vstack(list(map(np.ravel, parameter_grid))).T
    preps = map(xzpsi, params_stack)

    icos_results = par_preparations_classicality(preps, icos, ma, mb, ndetps_icos,
                                                 rounds_icos, workers, verb)
    np.save("heatmap-icos-grid", parameter_grid)
    np.save("heatmap-icos-alphas", icos_results)

    romb_results = par_preparations_classicality(preps, romb, ma, mb, ndetps_romb,
                                                 rounds_romb, workers, verb)
    np.save("heatmap-romb-grid", parameter_grid)
    np.save("heatmap-romb-alphas", icos_results)


if __name__ == "__main__":
    main()


