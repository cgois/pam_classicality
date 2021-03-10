import argparse
from itertools import combinations, chain, repeat
from functools import partial
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from random import sample

import numpy as np
from numpy import sin, cos
import picos
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from measurements import insphere_radius, romb
from measurements import mirror_symmetric as mirror_symm_meas
import detpoints as detps


WORKERS = 8

ZERO_TOL = 10E-7
ISCLOSE_RTOL = 1E-7
ISCLOSE_ATOL = 1E-7

N_ORDERINGS = int(1.2e3)
REDUND_ROUNDS = 50
N_THETAS = 30
MAX_THETA = np.pi / 2
PREPS = romb()
# print(f"preps. radius: {insphere_radius(PREPS)}")


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sample_detpoints(ma, mb, mx, my, nords):

    def independent_strategies(ma, mb, my):
        """Generate all single party strategies of length ma * my

        Method: enumerate n_lambdas integers and convert to a binary string with
        a suitable fixed length. Then convert to list of integers.
        """

        n_lambdas = mb ** (my * ma)
        detp = ['{:0{}b}'.format(el, ma * my) for el in range(n_lambdas)]
        detp = [[int(digit) for digit in el] for el in detp] # Convert binary str to list of ints.

        # Split a's in each strategy, e.g.: [0, 1, 1, 0] -> [[0, 1], [1, 0]] if ma = my = 2.
        return [[detp[i][j:j+my] for j in range(0, ma * my, my)] for i in range(n_lambdas)]


    indeps = independent_strategies(ma, mb, my)

    # Random message orderings
    random_ints = sample(range(0, 2 ** mx), nords)
    orderings = ['{:0{n}b}'.format(el, n=mx) for el in random_ints]
    orderings = [[int(digit) for digit in order] for order in orderings]

    detps = []
    for i in range(len(indeps)):
        for messages in orderings:
            detps.append([indeps[i][a] for a in messages])

    # Flatten the sublists and return unique strategies.
    detps = [list(chain(*detps[i])) for i in range(len(detps))]
    return np.asarray(detps).T # Takes a long time to get uniques, so I'm letting it be


def local_model(preparations, measurements, detpoints, verb=-1, solver="gurobi"):
    """Maximum visibility s.t. the measurement set has classical PAM model."""

    radius = insphere_radius(preparations)
    measurements, _ = np.split(measurements, 2)    # Ignore antipodals.
    assert np.all(measurements == -_), "Measurements were not correctly ordered."

    weights = picos.RealVariable("weights", detpoints.shape[1])
    d = picos.RealVariable("visibility", 1)
    dot = np.inner(preparations, measurements).flatten()
    behaviors = 0.5 * (1 + d * dot / radius)

    prob = picos.Problem()
    prob.add_list_of_constraints([d >= 0, d <= 1, weights >= 0, picos.sum(weights) == 1])
    prob.add_constraint(behaviors == detpoints * weights)    # * is matmul by type inference.
    prob.set_objective("max", d)

    prob.options.solver = solver
    prob.options.verbosity = verb
    prob.license_warnings = False

    return prob.solve()


def opt_local_model(preparations, measurements, detpoints, verb=-1, solver="gurobi"):
    """
    Implements searching for the best local model iteratively with the following steps:

    1. Find best model with starting strategies.
    2. Select the strategies with nonzero weights and discard the others.
    3. Complete the ones we kept with new randomly sampled strategies and run again.
    4. Break if optval == 1 or if the opt. values of last 'optrounds' rounds were equal (within tol.)
    """

    optvals = sample(range(-REDUND_ROUNDS, 0), REDUND_ROUNDS) # Store last self.optrounds optvals.
    round_count = 0

    # Break when optval is 1 or when all last optrounds are sufficiently close.
    while(optvals[-1] < 1 and not np.all(np.isclose(optvals, optvals[-1],
                                                     rtol=ISCLOSE_RTOL, atol=ISCLOSE_ATOL))):

        result = local_model(preparations, measurements, detpoints, verb, solver)

        # Discard zero-weighted strategies and sample new ones.
        weights = np.array(result.primals[list(result.primals.keys())[1]]) # TODO: Loose indexing.
        nonzeros = [i for i in range(len(weights)) if weights[i] >= ZERO_TOL]
        detpoints = np.c_[detpoints[:,nonzeros], sample_detpoints(2, 2, preparations.shape[0], 3, N_ORDERINGS)]

        # Append new optval and pop oldest one.
        optvals.append(result.value)
        optvals.pop(0)

        if verb >= 0:
            print(f'round {round_count} -> optval: ' + str(result.value) \
                  + f' / used_detps: {len(nonzeros)} / {detpoints.shape[1]}')
            round_count += 1

    assert detpoints[:,range(len(nonzeros))].shape[1] == weights[nonzeros].shape[0],\
    'nof. weights != nof. detpoints after optimizing local model!'

    return result.value


def main(verb=-1):

    # NOTE: I use only b0 in detps table, that's why I don't need meas. antipodals.
    thetas = np.linspace(0, MAX_THETA, N_THETAS)
    measurements = list(map(mirror_symm_meas, thetas))
    preparations = PREPS
    detpoints = sample_detpoints(2, 2, preparations.shape[0], 3, N_ORDERINGS)


    print("Starting...")
    print(f"preps. shape: {np.asarray(preparations).shape} / "
          f"meas. shape: {np.asarray(measurements).shape} / "
          f"detps shape: {np.asarray(detpoints).shape}")

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        alphas = list(tqdm(executor.map(opt_local_model,
                                        repeat(preparations),
                                        measurements,
                                        repeat(detpoints),
                                        repeat(verb)),
                           total=len(measurements)))

    return alphas, thetas


if __name__ == '__main__':

    alphas, thetas = main(verb=-1)
    basename = f"incompatibility-mirrorsymm-romb-nords{N_ORDERINGS}-rr{REDUND_ROUNDS}"
    np.save(basename + "-thetas", thetas)
    np.save(basename + "-alphas", alphas)
