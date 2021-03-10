from itertools import combinations, chain, repeat
import numpy as np
from numpy import sin, cos
import picos
from tqdm import tqdm

from measurements import insphere_radius, romb, icos
import detpoints as detps


MEAS = icos()
NX, NY = 50, 25
ZERO_TOL = 10E-7
ISCLOSE_RTOL = 1E-7
ISCLOSE_ATOL = 1E-7


def xzpsi(params):
    """Preparation with vectors [x, z, (r, theta, phi)]."""

    # phi -> azimuth (horizontal on plot), theta -> polar
    phi, theta = params
    return np.array([[1, 0, 0],
                     [0, 0, 1],
                     [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]])


def local_model(preparations, measurements, detpoints, verb=-1, solver="gurobi"):
    """Maximum visibility s.t. the measurement set has classical PAM model."""

    radius = insphere_radius(measurements)
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

    return prob.solve().value


def run():
    detpoints = np.asarray(detps.detpoints(2, 2, 3, MEAS.shape[0] // 2)).T
    parameter_grid = np.meshgrid(np.linspace(0, 2 * np.pi, NX), np.linspace(0, np.pi, NY))
    preparations = list(map(xzpsi, np.vstack(list(map(np.ravel, parameter_grid))).T))

    print("Starting...")
    print(f"preps. shape: {np.asarray(preparations).shape} / "
          f"meas. shape: {np.asarray(MEAS).shape} / "
          f"detps shape: {np.asarray(detpoints).shape}")


    results = list(tqdm(map(local_model,
                            preparations,
                            repeat(MEAS),
                            repeat(detpoints)),
                        total=len(preparations)))

    return parameter_grid, results
