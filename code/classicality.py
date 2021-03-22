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


def max_classical_measurement_visibility():
    pass


def max_classical_state_visibility(preps, meas, detpoints, verb=-1, solver="gurobi"):
    """Find best PAM-classical model for `preps` given `meas` and `detpoints`.

    Args:
        preps: list of ndarrays of density operators (all with same dimensions).
        meas: ordered list of effects (ndarrays) following, e.g. `[Z0, Z1, X0, X1]`.
        detpoints: row-wise deterministic strategies to consider.
        verb: solver verbosity level.
        solver: any LP solver accepted by `picos`.

    Returns:
        picos.Solution: its 'value' attribute is the maximum visibility s.t. a
            PAM-classical model exists, and in "primals" are the optimized weights
            for each strategy in `detpoints`.
    """

    dim = preps[0].shape[0]
    eta = insphere_radius([matrix2bloch(m) for m in meas])

    vis = pic.RealVariable("visibility", 1)
    weights = pic.RealVariable("weights", detpoints.shape[0])

    prob = pic.Problem()
    prob.set_objective("max", vis)
    prob.add_list_of_constraints([vis >= 0, vis <= 1, weights >= 0, pic.sum(weights) == 1])

    behaviors = []
    for i in range(len(preps)):
        o_x = (vis / eta) * preps[i] + (1 - vis / eta) * np.eye(dim) / dim
        for j in range(len(meas)):
            # [tr(O0 M(0|0)), tr(O1 M(0|0)), ..., tr(O0 M(1|0)), ..., tr(Omx M(mb|my)]
            behaviors.append(pic.trace(o_x * meas[j]))

    model = detpoints.T * weights
    prob.add_list_of_constraints([behaviors[i] == model[i] for i in range(len(behaviors))])

    prob.options.solver = solver
    prob.options.verbosity = verb
    prob.license_warnings = False

    return prob.solve()


def optimal_classical_model(preps, ma, meas, mb, ndetps, rounds, optfunc,
                             verb=-1, solver="gurobi"):
    """Iterative search for the best classical model iteratively.

    1. Find best model with self.ndetp starting strategies.
    2. Select the strategies with nonzero weights and discard the others.
    3. Complete the ones we kept with new randomly sampled strategies and run again.
    4. Break if optval = 1 or if the opt. values of last 'rounds' rounds were equal.

    Todo:
        * Documentation.
        * If take `meas` as, e.g., [[X0, X1], [Z0, Z1]] may infer mb.
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


if __name__ == '__main__':
    raise NotImplementedError
