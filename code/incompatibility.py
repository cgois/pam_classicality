"""Measurement incompatibility robustness SDPs
for 2 and 3 dichotomic measurements and white noise model.

Functions to make the plots of incompatibility in the paper.
"""

from itertools import chain, combinations

import numpy as np
from numpy import eye, sin, cos
import matplotlib.pyplot as plt
import seaborn as sns
import picos

from measurements import bloch2density
from measurements import mirror_symmetric as msym


def chunks(lst, n):
    """Split lst into n chunks."""

    return [lst[i:i + n] for i in range(0, len(lst), max(1, n))]


def incompatibility_robustness(*measurements, **kwargs):
    """Incompatibility robustness for the given measurements w.r.t. white noise.

    Args:
        *measurements: where each measurement is a list with its effects.
        **kwargs: can contain 'solver' (SDP) and 'verb' (verbosity) specification.

    Returns:
        picos.Solution: its 'value' attribute is the robustness and in 'primals'
            are the parent POVM operators.

    Example:
        >>> b0, b1 = np.array([1, 0]), np.array([0, 1])
        >>> Z0, X0 = np.outer(b0, b0), np.outer(b0 + b1, b0 + b1) / 2
        >>> Z1, X1 = np.eye(2) - Z0, np.eye(2) - X0
        >>> incompatibility_robustness([Z0, Z1], [X0, X1]).value
        0.7071067811559121

    Todo:
        * Generalize to measurements with more than `dim` effects
    """

    if "solver" not in kwargs:
        kwargs["solver"] = "cvxopt"
    if "verb" not in kwargs:
        kwargs["verb"] = 0
    dim = len(args[0])  # Only works for measurements with dim effects

    eta = picos.RealVariable("Robustness", 1)
    parent = [picos.HermitianVariable(f"G{i}", dim)
              for i in range(dim ** len(measurements))]  # Parent meas. effects

    prob = picos.Problem()
    prob.add_constraint(eta <= 1)
    prob.add_list_of_constraints([G >> 0 for G in parent])

    # Parent POVM constraints to reproduce measurements:
    block_size = 1
    for meas in measurements:
        view = chunks(parent, block_size)
        for oper in range(dim):
            parent_equiv = sum(chain(*view[oper::dim]))
            prob.add_constraint(parent_equiv ==
                                eta * meas[oper] + (1 - eta) * eye(dim) / dim)
        block_size *= dim

    prob.set_objective("max", eta)
    prob.options.solver = kwargs["solver"]
    prob.options.verbosity = kwargs["verb"]
    prob.license_warnings = False
    return prob.solve()



def robustness2(A1, B1, A2, B2, solver="cvxopt", verb=0):
    """Incompatibility robustness for two measurements and white noise."""

    eta = picos.RealVariable("Robustness", 1)
    G11 = picos.HermitianVariable("Parent POVM G11", 2)
    G12 = picos.HermitianVariable("Parent POVM G12", 2)
    G21 = picos.HermitianVariable("Parent POVM G21", 2)
    G22 = picos.HermitianVariable("Parent POVM G22", 2)

    prob = picos.Problem()
    prob.add_constraint(eta <= 1)
    prob.add_list_of_constraints([G11 >> 0, G12 >> 0, G21 >> 0, G22 >> 0])
    prob.add_constraint(G11 + G12 == eta * A1 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G21 + G22 == eta * A2 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G11 + G21 == eta * B1 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G12 + G22 == eta * B2 + (1 - eta) * np.eye(2) / 2)

    prob.set_objective("max", eta)
    prob.options.solver = solver
    prob.options.verbosity = verb
    prob.license_warnings = False

    # I think there's strong duality here, but anyway I'll
    # solve the dual just to make sure we get an upper bound.
    return prob.dual.solve()


def robustness3(A1, B1, C1, A2, B2, C2, solver="cvxopt", verb=0):
    """Incompatibility robustness for three measurements and white noise."""

    eta = picos.RealVariable("Robustness", 1)
    G111 = picos.HermitianVariable("Parent POVM G111", 2)
    G211 = picos.HermitianVariable("Parent POVM G211", 2)
    G121 = picos.HermitianVariable("Parent POVM G121", 2)
    G112 = picos.HermitianVariable("Parent POVM G112", 2)
    G221 = picos.HermitianVariable("Parent POVM G221", 2)
    G212 = picos.HermitianVariable("Parent POVM G212", 2)
    G122 = picos.HermitianVariable("Parent POVM G122", 2)
    G222 = picos.HermitianVariable("Parent POVM G222", 2)

    prob = picos.Problem()
    prob.add_constraint(eta <= 1)
    prob.add_list_of_constraints([G111 >> 0, G211 >> 0, G121 >> 0, G112 >> 0, G221 >> 0, G212 >> 0, G122 >> 0, G222 >> 0])
    prob.add_constraint(G111 + G112 + G121 + G122 == eta * A1 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G211 + G212 + G221 + G222 == eta * A2 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G111 + G112 + G211 + G212 == eta * B1 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G121 + G122 + G221 + G222 == eta * B2 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G111 + G121 + G211 + G221 == eta * C1 + (1 - eta) * np.eye(2) / 2)
    prob.add_constraint(G112 + G122 + G212 + G222 == eta * C2 + (1 - eta) * np.eye(2) / 2)

    prob.set_objective("max", eta)
    prob.options.solver = solver
    prob.options.verbosity = verb
    prob.license_warnings = False

    # I think there's strong duality here, but anyway I'll
    # solve the dual just to make sure we get an upper bound.
    return prob.dual.solve()


def mirror_symmetric_robustness(thetas):
    measurements = [[bloch2density(vec) for vec in msym(theta)] for theta in thetas]
    return [robustness3(*meas).value for meas in measurements]

def pairwise_robustness(thetas, bound):
    # Get a vector from each pair of elements then append their antipodals.
    pairs_idxs = combinations([0, 1, 2], 2)
    pairs_idxs = [pair + tuple(map(lambda x: x + 3, pair)) for pair in pairs_idxs]

    # Make the lists of pairs of measurements for each theta
    measurements = [[bloch2density(vec) for vec in msym(theta)] for theta in thetas]
    # pairs = [[meas[idx] for idx in pairs_idxs] for meas in measurements]
    return [bound([robustness2(*[meas[i] for i in pair]).value for pair in pairs_idxs])
            for meas in measurements]


# def paper_robustness(thetas):
#     def measurements_paper(theta):
#         """Eq. 20 of Designolle et al. "Incompatibility robustness of quantum measurements",
#         to reproduce fig. 4 of the same paper using robustness2 function."""

#         X = np.array([[0, 1], [1, 0]])
#         Z = np.array([[1, 0], [0, -1]])
#         A1 = (1 / 2) * (np.eye(2) + cos(theta) * Z + sin(theta) * X)
#         A2 = (1 / 2) * (np.eye(2) - cos(theta) * Z - sin(theta) * X)
#         B1 = (1 / 2) * (np.eye(2) + cos(theta) * Z - sin(theta) * X)
#         B2 = (1 / 2) * (np.eye(2) - cos(theta) * Z + sin(theta) * X)
#         return A1, B1, A2, B2

#     return [robustness2(*measurements_paper(theta)).value for theta in thetas]


def plot_msym_robustness_with_classicality(class_thetas, class_alphas):

    sns.set(style='whitegrid')
    sns.set_context("paper", font_scale=1.6)
    plt.figure(figsize=(8,6))
    palette = sns.color_palette("Paired")

    robustness = mirror_symmetric_robustness(class_thetas)
    pair_robustness_max = pairwise_robustness(class_thetas, max)
    pair_robustness_min = pairwise_robustness(class_thetas, min)

    plt.scatter(class_thetas, pair_robustness_min, label="Pairwise incompatibility robustness (min)", color=palette[5], alpha=.5)
    plt.scatter(class_thetas, pair_robustness_max, label="Pairwise incompatibility robustness (max)", color=palette[3], alpha=.5)
    plt.scatter(class_thetas, class_alphas, label="Measurement classicality lower bound", color=palette[1])
    plt.scatter(class_thetas, robustness, label="Incompatibility robustness", color=palette[9], alpha=1)
    # plt.fill_between(class_thetas, class_alphas, robustness,
                     # where=class_alphas > robustness, alpha=.5, interpolate=True, color=palette[0])

    plt.xlim([0, np.pi / 2])
    plt.ylabel(r"$\alpha, \,\chi$")
    plt.xlabel(r"$\theta$ (rad)")
    plt.legend()
    plt.tight_layout()
