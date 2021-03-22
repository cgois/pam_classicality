import numpy as np
from numpy import sin, cos
from itertools import combinations, chain, repeat
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from pam.classicality import par_preparations_classicality
from pam.utils import mirror_symmetric, bloch2matrix, incompatibility_robustness, romb, icos, chunks


def bloom_triads(theta):
    """All 3-wise combinations of preparations described in fig. 4."""

    states = np.asarray([(cos(theta + np.pi / 2), sin(theta + np.pi / 2), 0),
                         ( cos(-theta + np.pi / 2), sin(-theta + np.pi / 2), 0),
                         (0, sin(theta + np.pi / 2), cos(theta + np.pi / 2)),
                         (0, sin(-theta + np.pi / 2), cos(-theta + np.pi / 2))])
    return [states[list(i)] for i in combinations(range(4), 3)]


def plot(thetas, alphas):
    """Plot violation curve for 4 preps. vs. classicality curve for all triads (fig. 5)."""

    sns.set(style='whitegrid')
    sns.set_context("paper", font_scale=1.6)
    plt.figure(figsize=(8,6))

    palette = sns.color_palette("Paired")
    ys2 = 1 / (np.sqrt(2) * np.sin(thetas))  # Violation curve.
    plt.scatter(thetas, alphas, label="Classicality lower bound", color=palette[1])
    plt.plot(thetas, ys2, label=r"$S=4$", color=palette[9], alpha=.5, linewidth=2)
    plt.fill_between(thetas, alphas, ys2, where=alphas > ys2, alpha=.5,
                     interpolate=True, color=palette[0])

    plt.ylim([0.7, 1])
    plt.xlim([0, np.pi / 2])
    plt.ylabel(r"$\alpha$")
    plt.xlabel(r"$\theta$ (rad)")
    plt.legend()
    plt.tight_layout()


def main():

    # Testing params:
    # nthetas = 20
    # meas = icos()
    # ndetps = int(1e3)
    # rounds = 5
    # workers = 5

    # Working params:
    nthetas = 50
    meas = romb()
    ndetps = int(.5e5)
    rounds = 50
    workers = 8
    verb = -1
    solver = "gurobi"

    print(f"workers={workers}, nthetas={nthetas}, ndetps={ndetps}, "\
          f"meassize={meas.shape}, redrounds={rounds}")

    thetas = np.linspace(0, np.pi / 2, nthetas)
    preps = list(chain(*map(bloom_triads, thetas)))
    ma, mb = 2, 2

    results = par_preparations_classicality(preps, meas, ma, mb, ndetps,
                                            rounds, workers, verb, solver)
    
    # Separate results corresponding to each theta and get lowest visibility of each.
    results = chunks(results, len(preps) // nthetas)
    minoptvals = []
    for triads in results:
        minoptvals.append(min([triad for triad in triads]))

    np.save('./activation-thetas', thetas)
    np.save('./activation-alphas', minoptvals)


if __name__ == '__main__':
    main()

