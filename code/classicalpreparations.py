from random import sample, randrange, randint
from typing import Union, List, Dict
import numpy as np
import picos

from measurements import insphere_radius
from scenario import LocalModel



class LPM(LocalModel):

    def __init__(self, measurements: np.ndarray,
                 ndetp: int,
                 optrounds: int = 1,
                 mx: int = 3,
                 ma: int = 2,
                 meastag: str = "untagged_measurement"):

        """Streamline of LPM instantiation with some default parameters."""

        # WARNING: mb != 2 is not implemented! (only proj. measurements on qbits)
        scenario = {"mx": mx, "ma": ma, "my": measurements.shape[0] // 2, "mb": 2}
        super().__init__(measurements, scenario, ndetp, optrounds, meastag)


    def det_points(self, ndetp: int, scenario: Dict[str, int]) -> np.ndarray:
        """Build a table with ndetp randomly sampled extremal points from the LPM polytope.
        We suppose Alice is the preparation and Bob the (dichotomic) measurements.

        Procedure: to generate one det. point, generate an ma*my length list of random bits,
        then repeat mx-ma blocks of it at random positions.
        """

        def independent_strategies(ndetp: int,
                                   scenario: Dict[str, int]) -> np.ndarray:
            """Generate a ma*my length list of random bits. (Corresponds to sampling
            ndetp deterministic strategies for a given nof. meas. and results.)

            A list of ndetp unique integers in the given range is created.
            (Except if n_lambdas becomes larger than long, in which case uniqueness is dropped).

            Each sampled decimal is cast to a binary string with a suitable fixed length.
            We then split digits of each string in the list and cast them as an integer.
            """

            n_lambdas = scenario['mb'] ** (scenario['my'] * scenario['ma'])

            try:
                random_ints = sample(range(0, n_lambdas), ndetp)
            except OverflowError: # Can't use unique sampling if n_lambdas is too large.
                random_ints = [randint(0, n_lambdas) for _ in range(ndetp)]

            detp = ['{:0{}b}'.format(el, scenario['ma'] * scenario['my'])
                        for el in random_ints]

            return np.array([[int(digit) for digit in el] for el in detp]).T

        # --------------------------

        # Sample ndetp independent strategies and repeat mx-ma blocks:
        ind_strs = independent_strategies(ndetp, scenario)
        detps = ind_strs
        my = scenario['my']

        # NOTE: I'm not picking a different "a" for each strategy, but rather repeating
        # the whole block at once (effectively: picking the same "a" and repeating at same pos.
        # for all strategies of the current round. I did it independently before and found no
        # difference in the results when using many redundant rounds.
        for _ in range(scenario['mx'] - scenario['ma']): # Nof. repeated blocks.
            ins_position = randrange(1 + detps.shape[0] // my) * my # Position for repeated block.
            rep_str = randrange(scenario['ma']) # Which "a" strategy to repeat.
            detps = np.insert(detps, ins_position,
                              ind_strs[rep_str * my:(rep_str + 1) * my, :], axis=0)

        return detps


    def find_model(self, state: np.ndarray,
                   detpoints: np.ndarray,
                   verb: int = -1,
                   solver: str = "gurobi") -> picos.modeling.solution.Solution:
        """Maximum visibility such that preparation 'state' has an LPM model for given strategies."""

        radius = insphere_radius(self.measurements)
        meas, _ = np.split(self.measurements, 2)    # Ignore antipodals.
        assert np.all(meas == -_), "Measurements were not correctly ordered."

        weights = picos.RealVariable("weights", detpoints.shape[1])
        d = picos.RealVariable("visibility", 1)
        dot = np.inner(state, meas).flatten()
        behaviors = 0.5 * (1 + d * dot / radius)

        prob = picos.Problem()
        prob.add_list_of_constraints([d >= 0, d <= 1, weights >= 0, picos.sum(weights) == 1])
        prob.add_constraint(behaviors == detpoints * weights)    # * is matmul by type inference.
        prob.set_objective("max", d)

        prob.options.solver = solver
        prob.options.verbosity = verb
        prob.license_warnings = False

        return prob.solve()



# TODO: Reimplementar chamada pelo terminal
if __name__ == '__main__':
    pass
