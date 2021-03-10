from __future__ import annotations
from collections import namedtuple
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from random import sample

import numpy as np
import picos
from tqdm import tqdm

from measurements import insphere_radius as inspr
from results import Result, Results


ZERO_TOL = 10E-8


@dataclass(frozen=True) # To allow mutation we'd have to keep track of updating detpoints...
class Scenario():
    """Store and maintain scenario parameters."""

    measurements: np.ndarray
    scenario: Dict[str, int]
    ndetp: int
    detpoints: np.ndarray
    optrounds: int = 10
    meastag: str = "untagged measurement"

    def __str__(self):
        return (f"Scenario: {self.scenario}\n"
                f"Measurements: {self.meastag} with radius = {inspr(self.measurements):.3f}\n"
                f"Strategies per round: {self.ndetp}\n"
                f"Redundant rounds: {self.optrounds}\n")



class LocalModel(Scenario, ABC):
    """Base class defining a framework for finding local models in arbitrary scenarios.

    Any LocalModel is a (locality) Scenario. It implements common generic procedures,
    such as applying a local model search to a list of states, filtering which deterministic
    strategies were used in a solution, do the search in multiple rounds so we can use
    larger measurements, and fully paralelize the application to multiple states.

    Any subclass of this ABC must specify a locality model by implementing:
    - det_points: a function that returns 'ndetp' deterministic points for a given 'scenario'.
    - find_model: a function that optimizes the model for a given state and detpoints.

    We can, for instance, define a LHV(LocalModel) class, and by implementing only those two
    methods it'll inherit advanced functionality from LocalModel.
    """

    def __init__(self, measurements: np.ndarray,
                 scenario: Dict[str, int],
                 ndetp: int, optrounds: int = 1,
                 meastag: str = "untagged_measurement"):

        super().__init__(measurements, scenario, ndetp, self.det_points(ndetp, scenario), optrounds, meastag)



    @abstractmethod
    def det_points(self, ndetp: int, scenario: Dict[str, int]) -> Union[np.ndarray, List]:
        return NotImplemented


    @abstractmethod
    def find_model(self, state: np.ndarray,
                   detpoints: np.ndarray,
                   verb: int = -1,
                   solver: str = "gurobi") -> picos.modeling.solution.Solution:
        return NotImplemented



    # TODO: Essas local_model e optimal_local_model não ficam melhores como decorador da find_model?
    def local_model(self, states: Union[List[np.ndarray], np.ndarray],
                    parameters: Optional[Union[List, np.ndarray]] = None,
                    description: str = '',
                    workers: int = 1,
                    verb: int = -1,
                    solver: str = "gurobi") -> Union[Result, Results]:
        """Find the optimal local model for each of 'states'.

        If states is an np.ndarray, it's considered to be a single state. Otherwise (if it's a list,
        a map etc.) we iterate over it and solve the model for each state.

        parameters: may be parameters related to each optval, like a meshgrid (helps plotting).
        description: may take any string with description of the results.
        workers: nof. workers for parallelization.
        verb: verbosity level in [-1, 0, 1] meaning nothing, method and method + solver output.
       """

        full_description = (f'Measurements: {self.meastag}'
                            f'\nDescription: {str(description)}'
                            f'\nDet. points per round: {self.ndetp}'
                            f'\nRedundant rounds: {self.optrounds}')
        results = Results(self.measurements, parameters, full_description, [])

        # TODO: Return Result instance instead.
        if isinstance(states, np.ndarray): # Single state.
            return self.optimal_local_model(states, verb, solver)

        elif workers == 1: # Multiple states with serial application.
            [results.append(self.optimal_local_model(state, verb, solver)) for state in states]

        else:
            states = list(states) # In case states was an iterator, as we need len for tqdm progress bar.
            with ProcessPoolExecutor(workers) as exec:
                mapres = list(tqdm(exec.map(partial(self.optimal_local_model, verb=verb, solver=solver),
                                            states), total=len(states)))
            [results.append(res) for res in mapres]

        return results


    def optimal_local_model(self, state: np.ndarray, verb: int = -1, solver: str = "gurobi"):
        """
        Implements searching for the best local model iteratively with the following steps:

        1. Find best model with self.ndetp starting strategies.
        2. Select the strategies with nonzero weights and discard the others.
        3. Complete the ones we kept with new randomly sampled strategies and run again.
        4. Break if optval = 1 or if the opt. values of last 'optrounds' rounds were equal (within tol.)
        """

        ISCLOSE_RTOL = 1E-7
        ISCLOSE_ATOL = 1E-7

        if verb >= 0:
            print(f"\n{'-' * 30}\nstate:\n{state}\n")

        optvals = sample(range(-self.optrounds, 0), self.optrounds) # Store last self.optrounds optvals.
        detpoints = self.detpoints
        round_count = 0

        # Break when optval is 1 or when all last optrounds are sufficiently close.
        while(optvals[-1] != 1 and not np.all(np.isclose(optvals, optvals[-1],
                                                         rtol=ISCLOSE_RTOL, atol=ISCLOSE_ATOL))):

            result = self.find_model(state, detpoints, verb, solver)

            # Discard zero-weighted strategies and sample new ones.
            weights = np.array(result.primals[list(result.primals.keys())[1]]) # TODO: Loose indexing.
            nonzeros = [i for i in range(len(weights)) if weights[i] >= ZERO_TOL]
            detpoints = np.c_[detpoints[:,nonzeros], self.det_points(self.ndetp - len(nonzeros), self.scenario)]

            # Append new optval and pop oldest one.
            optvals.append(result.value)
            optvals.pop(0)

            if verb >= 0:
                print(f'round {round_count} -> optval: ' + str(result.value))
                round_count += 1

        assert detpoints[:,range(len(nonzeros))].shape[1] == weights[nonzeros].shape[0],\
        'nof. weights != nof. detpoints after optimizing local model!'

        return Result(state, detpoints[:,range(len(nonzeros))], weights[nonzeros], result.value)
