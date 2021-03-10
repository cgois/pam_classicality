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


def xzpsi(params):
    """Preparation with vectors [x, z, (r, theta, phi)]."""

    # phi -> azimuth (horizontal on plot), theta -> polar
    phi, theta = params
    return np.array([[1, 0, 0],
                     [0, 0, 1],
                     [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]])


if __name__ == "__main__":

    nx, ny = 80, 40
    parameter_grid = np.meshgrid(np.linspace(0, 2 * np.pi, nx),
                                 np.linspace(0, np.pi, ny))
    params_stack = np.vstack(list(map(np.ravel, parameter_grid))).T
    preparations = map(xzpsi, params_stack)

    scenario = lpm.LPM(MEAS, NDETPS, REDROUNDS, meastag='icosahedron')
    description = (f"xzpsi states for {nx} x {ny} grid of phis and thetas.")

    results = scenario.local_model(preparations, parameters=parameter_grid,
                                   description=description, workers=WORKERS, verb=-1)
    results.save('xzpsi-icos')
