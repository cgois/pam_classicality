"""
Measurement classicality vs. incompatibility.
Corresponds to results presented in sec. IV.A.
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pam.classicality import par_measurements_classicality
from pam.utils import mirror_symmetric, bloch2matrix, incompatibility_robustness, romb
from pam.classicality import measurements_classicality


def classicality(preps, meas, ma, mb, ndetps, rounds, workers, verb, solver):
    """Measurement classicality computation."""

    print(f"workers={workers}, ndetps={ndetps}, preps.shape={preps.shape}, "\
          f"meas.shape={np.asarray(meas).shape}, redrounds={rounds}")

    return par_measurements_classicality(preps, meas, ma, mb, ndetps,
                                         rounds, workers, verb, solver)


def robustness(thetas):
    """Incompatibility robustness for the mirror symmetric measurements."""

    optvals = []
    for theta in thetas:
        meas = [bloch2matrix(eff) for eff in mirror_symmetric(theta)]
        optvals.append(incompatibility_robustness(meas[0:2], meas[2:4], meas[4:6]).value)
    return optvals


def plot(class_thetas, class_alphas, robust_thetas, robust_alphas):

    sns.set(style='whitegrid')
    sns.set_context("paper", font_scale=1.6)
    plt.figure(figsize=(8,6))
    palette = sns.color_palette("Paired")

    plt.scatter(class_thetas, class_alphas,
                label="Measurement classicality lower bound", color=palette[1])
    plt.scatter(robust_thetas, robust_alphas,
                label="Incompatibility robustness", color=palette[9], alpha=1)
    plt.fill_between(class_thetas, class_alphas, robust_alphas,
                     where=class_alphas > robust_alphas, alpha=.5,
                     interpolate=True, color=palette[0])

    plt.xlim([min(class_thetas), max(class_thetas)])
    plt.ylabel(r"$\alpha, \,\chi$")
    plt.xlabel(r"$\theta$ (rad)")
    plt.legend()
    plt.tight_layout()


def main(verb=-1):

    # Testing params:
    # nthetas = 20
    # max_theta = np.pi / 2
    # ndetps = int(1e4)
    # rounds = 10
    # workers = 8
    # solver = "gurobi"

    # Working params:
    nthetas = 50
    max_theta = np.pi / 2
    ndetps = int(.5e5)
    rounds = 50
    workers = 10
    solver = "gurobi"

    preps = romb()
    ma, mb = 2, 2
    thetas = np.linspace(0, max_theta, nthetas)

    chis = robustness(thetas)
    basename = f"./robustness-msym-"
    np.save(basename + "thetas", thetas)
    np.save(basename + "chis", chis)

    meas = list(map(mirror_symmetric, thetas))
    alphas = classicality(preps, meas, ma, mb, ndetps, rounds, workers, verb, solver)
    basename = f"./class-msym-"
    np.save(basename + "thetas", thetas)
    np.save(basename + "alphas", alphas)


if __name__ == '__main__':
    main()