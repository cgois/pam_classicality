from itertools import chain
from random import sample, randrange, randint
from typing import Union, List, Dict

import numpy as np
import picos as pic

from detpoints import build_detpoints
from utils import insphere_radius, matrix2bloch


RTOL = 1E-7
ATOL = 1E-7
ZERO_TOL = 1E-6
SEPARATOR = "-" * 40


def max_classical_preps_visibility(preps, meas, detpoints, verb=-1, solver="gurobi"):
    """Find best PAM-classical model for `preps` given `meas` and `detpoints`.

    Ref.: Program 6 in https://arxiv.org/abs/2101.10459

    This function does not check for valid preparations and measurements.
    You must make sure all Bloch vectors in `preps` are valid, with all the same
    dimensions, and likewise for the ones in `meas`.

    Args:
        preps (ndarray): row-wise Bloch vectors of density operators.
        meas (ndarray): Bloch vectors of measurement effects following. May be either
            given as ordered row-wise vectors e.g. `[Z0, Z1, X0, X1]` or nested as
            e.g. `[[Z0, Z1], [X0, X1]]`.
        detpoints: row-wise deterministic strategies to consider.
        verb: solver verbosity level.
        solver: any LP solver accepted by `picos`.

    Returns:
        picos.Solution: its 'value' attribute is the maximum visibility s.t. a
            PAM-classical model exists, and in "primals" are the optimized weights
            for each strategy in `detpoints`.
    """

    if meas.ndim == 3:  # Flatten measurements if nested.
        meas = np.asarray(list(chain(*meas)))
    dim = preps[0].shape[0]
    eta = insphere_radius(meas)

    vis = pic.RealVariable("visibility", 1)
    weights = pic.RealVariable("weights", detpoints.shape[0])

    prob = pic.Problem()
    prob.set_objective("max", vis)
    prob.add_list_of_constraints([vis >= 0, vis <= 1, weights >= 0, pic.sum(weights) == 1])

    # Classical preparations model constraint:
    dot = np.inner(preps, meas).flatten()
    behaviors = 1 / dim + (1 / 2) * vis * dot / eta
    prob.add_constraint(behaviors == detpoints.T * weights)

    prob.options.solver = solver
    prob.options.verbosity = verb
    prob.license_warnings = False

    return prob.solve()


def max_classical_meas_visibility(preps, meas, detpoints, verb=-1, solver="gurobi"):
    """Find best PAM-classical model for `meas` given `preps` and `detpoints`.

    Ref.: Maximization version of program 9 in https://arxiv.org/abs/2101.10459

    This function does not check for valid preparations and measurements.
    You must make sure all Bloch vectors in `preps` are valid, with all the same
    dimensions, and likewise for the ones in `meas`.

    Args:
        preps (ndarray): row-wise Bloch vectors of density operators.
        meas (ndarray): Bloch vectors of measurement effects following. May be either
            given as ordered row-wise vectors e.g. `[Z0, Z1, X0, X1]` or nested as
            e.g. `[[Z0, Z1], [X0, X1]]`.
        detpoints (ndarray): row-wise deterministic strategies to consider.
        verb (int): solver verbosity level in [-1, 0, 1].
        solver (string): any LP solver accepted by `picos`.

    Returns:
        picos.Solution: its 'value' attribute is the maximum visibility s.t. a
            PAM-classical model exists, and in "primals" are the optimized weights
            for each strategy in `detpoints`.
    """

    if meas.ndim == 3:  # Flatten measurements if nested.
        meas = np.asarray(list(chain(*meas)))
    dim = preps[0].shape[0]
    eta = insphere_radius(preparations)

    vis = pic.RealVariable("visibility", 1)
    weights = picos.RealVariable("weights", detpoints.shape[0])

    prob = pic.Problem()
    prob.set_objective("max", vis)
    prob.add_list_of_constraints([vis >= 0, vis <= 1, weights >= 0, pic.sum(weights) == 1])

    # Classical measurement model constraint:
    dot = np.inner(preps, meas).flatten()
    behaviors = 1 / dim + (1 / 2) * vis * dot / eta
    prob.add_constraint(behaviors == detpoints.T * weights)

    prob.options.solver = solver
    prob.options.verbosity = verb
    prob.license_warnings = False

    return prob.solve()


def optimal_classical_model(preps, meas, ma, mb, ndetps, rounds, optfunc,
                             verb=-1, solver="gurobi"):
    """Iterative search for the best classical model iteratively.

    1. Find best model through `optfunc` with `ndetp` starting strategies.
    2. Select the strategies with nonzero weights and discard the others.
    3. Complete the ones we kept with new randomly sampled strategies and run again.
    4. Break if optval = 1 or if the opt. values of last 'rounds' rounds were equal.

    Args:
        preps (ndarray): row-wise Bloch vectors of density operators.
        ma (int): classical preparations dimension.
        meas (ndarray): Bloch vectors of measurement effects following. May be either
            given as ordered row-wise vectors e.g. `[Z0, Z1, X0, X1]` or nested as
            e.g. `[[Z0, Z1], [X0, X1]]`.
        mb (int): nof. effects per measurement.
        ndetps (int): nof. deterministic strategies to use in each round.
        rounds (int): nof. rounds with no improvement in model before breaking.
        optfunc (function): function that optimizes the model.
        verb (int): solver verbosity level in [-1, 0, 1].
        solver (string): any LP solver accepted by `picos`.

    Returns:
        picos.Solution: its 'value' attribute is the maximum visibility s.t. a
            PAM-classical model exists, and in "primals" are the optimized weights
            for each strategy in `detpoints`.
    """

    mx, my = len(preps), len(meas) // mb
    detpoints = build_detpoints(ma, mb, mx, my, samples=ndetps)
    optvals = sample(range(-rounds, 0), rounds) # Store `rounds` previous vals.
    round_count = 0

    if verb >= 0:
        print(SEPARATOR + "\nOptimizing classical model:\n")

    # Break when optval is 1 or when all last rounds are sufficiently close.
    while(optvals[-1] != 1 and not
          np.all(np.isclose(optvals, optvals[-1], rtol=RTOL, atol=ATOL))):

        result = optfunc(preps, meas, detpoints, verb, solver)
        optvals.append(result.value)
        optvals.pop(0)

        # Discard zero-weighted strategies and sample new ones.
        weights = np.array(result.primals[list(result.primals.keys())[1]])
        nonzeros = [i for i in range(len(weights)) if weights[i] >= ZERO_TOL]
        detpoints = np.vstack((detpoints[nonzeros,:],
                               build_detpoints(ma, mb, mx, my, ndetps - len(nonzeros))))

        if verb >= 0:
            print(f'round {round_count} -> optval: ' + str(result.value))
            round_count += 1

    assert detpoints[:,range(len(nonzeros))].shape[1] == weights[nonzeros].shape[0],\
    'nof. weights != nof. detpoints after optimizing local model!'

    return result


def preparations_classicality(preps, meas, ma, mb, ndetps, rounds, verb=-1):
    """Specify `optimal_classical_model` to preparations classicality."""

    return optimal_classical_model(preps, meas, ma, mb, ndetps, rounds,
                                   optfunc=max_classical_preps_visibility, verb=verb)

def measurements_classicality(preps, meas, ma, mb, ndetps, rounds, verb=-1):
    """Specify `optimal_classical_model` to measurements classicality."""
    
    return optimal_classical_model(preps, meas, ma, mb, ndetps, rounds,
                                   optfunc=max_classical_meas_visibility, verb=verb)


if __name__ == '__main__':
    raise NotImplementedError
