import argparse

import numpy as np
from numpy import sin, cos
from itertools import combinations
from functools import partial
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import classicalpreparations as lpm
import measurements


N_THETAS = 50
MEAS = measurements.romb()
NDETPS = int(1e5)
REDROUNDS = 70
WORKERS = 10


def bloom_triads(theta):
    """Return all triads for the given theta."""

    states = np.asarray([(cos(theta + np.pi / 2), sin(theta + np.pi / 2), 0),
                         ( cos(-theta + np.pi / 2), sin(-theta + np.pi / 2), 0),
                         (0, sin(theta + np.pi / 2), cos(theta + np.pi / 2)),
                         (0, sin(-theta + np.pi / 2), cos(-theta + np.pi / 2))])
    return [states[list(i)] for i in combinations(range(4), 3)]


def activation_plot(thetas, alphas):

    sns.set(style='whitegrid')
    sns.set_context("paper", font_scale=1.6)
    plt.figure(figsize=(8,6))

    palette = sns.color_palette("Paired")
    ys2 = 1 / (np.sqrt(2) * np.sin(thetas))
    # plt.scatter(thetas, alphas, label="Classicality lower bound", color=sns.color_palette("tab10")[0])#"#0b9bb8")
    # plt.plot(thetas, ys2, color=palette[4], label=r"$S=4$")
    # plt.fill_between(thetas, alphas, ys2, where=alphas > ys2, alpha=.5, interpolate=True, color="#b1abd9")
    plt.scatter(thetas, alphas, label="Classicality lower bound", color=palette[1])
    plt.plot(thetas, ys2, label=r"$S=4$", color=palette[9], alpha=.5, linewidth=2)
    plt.fill_between(thetas, alphas, ys2, where=alphas > ys2, alpha=.5, interpolate=True, color=palette[0])


    plt.ylim([0.7, 1])
    plt.xlim([0, np.pi / 2])
    plt.ylabel(r"$\alpha$")
    plt.xlabel(r"$\theta$ (rad)")
    plt.legend()
    plt.tight_layout()


if __name__ == '__main__':

    print(f'workers={WORKERS}, nthetas={N_THETAS}, ndetps={NDETPS}, meassize={MEAS.shape}, redrounds={REDROUNDS}')

    scenario = lpm.LPM(MEAS, NDETPS, REDROUNDS, meastag='rombicuboctahedron')
    thetas = np.linspace(0, np.pi / 2, N_THETAS)
    preparations = map(bloom_triads, thetas)

    # I'll keep the 4-list of triads and parallelize while using local_model in serial
    # as It's more readable and easier to debug in this case.
    with ProcessPoolExecutor(WORKERS) as exec:
        results = list(tqdm(exec.map(partial(scenario.local_model,
                                             description=f'bloom states parametrized in theta\nredrounds={REDROUNDS}',
                                             verb=-1,
                                             workers=1),
                                     preparations),
                            total=len(thetas)))

    # Export full results for each set of triads, and also
    # optvals s.t. every triad for a given theta has a local model.
    minoptvals = []
    for i in range(len(results)):
        results[i].parameters = thetas[i]
        results[i].save(f'bloom_triads-theta={thetas[i]:.3f}')
        minoptvals.append(min(results[i].optvals))

    np.save('bloom-thetas', thetas)
    np.save('bloom-alphas', minoptvals)
