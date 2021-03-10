import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

import lpm
import measurements


MEAS = measurements.icos()
NDETPS = 700
REDROUNDS = 100
WORKERS = 10
N_POLARS, N_AZIMUTHS = 15, 30


def sphere(params):
    """Preparation with vectors [z, (alpha, beta), (theta, phi)]."""

    # beta/phi -> azimuth, alpha/theta -> polar
    alpha, beta, phi, theta = params
    return np.array([[0, 0, 1],
                     [sin(alpha) * cos(beta), sin(alpha) * sin(beta), cos(alpha)],
                     [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]])


if __name__ == "__main__":

    polars = np.linspace(0, np.pi, N_POLARS)
    azimuths = np.linspace(0, 2 * np.pi, N_AZIMUTHS)
    parameter_grid = np.meshgrid(polars, azimuths, polars, azimuths)
    parameter_stack = np.vstack(list(map(np.ravel, parameter_grid))).T
    preparations = map(sphere, parameter_stack)

    scenario = lpm.LPM(MEAS, NDETPS, REDROUNDS, meastag='icosahedron')
    description = (f"spherical coordinates [z, (alpha, beta), (theta, phi)] states for"
                   f"{N_POLARS} x {N_AZIMUTHS} grid of polars and azimuths.")

    results = scenario.local_model(preparations, parameters=parameter_stack,
                                   description=description, workers=WORKERS, verb=-1)
    results.save('sphere-icos')
