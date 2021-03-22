"""
Measurement classicality vs. incompatibility.
Corresponds to results presented in sec. IV.A.
"""

from concurrent.futures import ProcessPoolExecutor

import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import ..code as pam
from ..code import mirror_symmetric, bloch2matrix, incompatibility_robustness


WORKERS = 8
N_ROUNDS = 50
N_DETPS = int(1e5)
N_THETAS = 30
MAX_THETA = np.pi / 2


def classicality(preps, meas, ma, mb, ndetps, rounds, verb=-1):
    """Measurement classicality computation."""

    if verb >= 0:
        print("Starting...")
        print(f"preps. shape: {np.asarray(preparations).shape} / "
              f"meas. shape: {np.asarray(measurements).shape}")

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        results = list(tqdm(executor.map(pam.classicality.measurements_classicality,
                                        repeat(preps),
                                        meas,
                                        repeat(ma),
                                        repeat(mb),
                                        repeat(ndetps),
                                        repeat(nrounds)
                                        repeat(verb)),
                           total=len(measurements)))
    return [result.value for result in results]


def robustness(thetas):
    """Incompatibility robustness for the mirror symmetric measurements."""

    optvals = []
    for theta in thetas:
        meas = [bloch2matrix(eff) for eff in mirror_symmetric(theta)]
        optvals.append(incompatibility_robustness(meas[0:2], meas[2:4], meas[4:6])
    return optvals


def plot(class_thetas, class_alphas, robust_thetas, robust_alphas):

    sns.set(style='whitegrid')
    sns.set_context("paper", font_scale=1.6)
    plt.figure(figsize=(8,6))
    palette = sns.color_palette("Paired")

    robustness = mirror_symmetric_robustness(class_thetas)
    pair_robustness_max = pairwise_robustness(class_thetas, max)
    pair_robustness_min = pairwise_robustness(class_thetas, min)

    plt.scatter(class_thetas, class_alphas,
                label="Measurement classicality lower bound", color=palette[1])
    plt.scatter(robust_thetas, robust_alphas,
                label="Incompatibility robustness", color=palette[9], alpha=1)
    plt.fill_between(class_thetas, class_alphas, robust_alphas,
                     where=class_alphas > robustness, alpha=.5,
                     interpolate=True, color=palette[0])

    plt.xlim([min(class_thetas), max(class_thetas)])
    plt.ylabel(r"$\alpha, \,\chi$")
    plt.xlabel(r"$\theta$ (rad)")
    plt.legend()
    plt.tight_layout()


def main():
    preps = pam.utils.romb()
    thetas = np.linspace(0, MAX_THETA, N_THETAS)

    chis = robustness(thetas)
    basename = f"robustness-msymm"
    np.save(basename + "-thetas", thetas)
    np.save(basename + "-chis", chis)

    meas = list(map(pam.utils.mirror_symmmetric, thetas))
    alphas = classicality(preps, meas, 2, 2, N_DETPS, N_ROUNDS, verb=-1)
    basename = f"incompat-msymm-romb-ndetps{ndetps}-rr{rounds}"
    np.save(basename + "-thetas", thetas)
    np.save(basename + "-alphas", alphas)


if __name__ == '__main__':
    main()