# TODO: Documentatation.


import time
import functools
from itertools import product, chain

import numpy as np
from numpy import sin, cos
import picos
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


"""Decorators and generalities"""

def chunks(lst, n):
    """Split lst into n chunks."""

    return [lst[i:i + n] for i in range(0, len(lst), max(1, n))]


def hemispherectomy(func):
    """Removes all southern hemisphere vectors."""

    @functools.wraps(func)
    def hemispherectomy_wrapper(*args):
        coords = func(*args)
        negs = [r for r in range(coords.shape[0]) if coords[r, -1] < 0]
        coords = np.delete(coords, negs, axis=0)

        return coords
    return hemispherectomy_wrapper


def antipodals(func):
    """Takes a function that generates vectors and appends their antipodals
    to the result.

    The antipodals will follow the same ordering in the second half
    of the returned list of values."""

    @functools.wraps(func)
    def antipodals_wrapper(*args):
        coords = func(*args)
        antipods = -coords

        return np.r_[coords, antipods]
    return antipodals_wrapper


def normalize(func):
    """Make an array generating function return it with normalized rows."""

    @functools.wraps(func)
    def normalize_wrapper(*args):
        meas = func(*args)
        norm = np.sum(np.square(meas), 1) ** 0.5

        return (meas.T / norm).T
    return normalize_wrapper


def timeit(func):
    """Prints running time of func."""

    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        #print(f"{func.__name__!r} took {end - start:.3f}s to run.")
        return result, end - start

    return timeit_wrapper


"""Operator representations"""

def gell_mann_matrix(j, k, dim=2):
    """Generalized Gell-Mann matrix with indexes j and k for given dimension.

    j < k: symmetric Gell-Mann matrices.
    j > k: anti-symmetric Gell-Mann matrices.
    j == k: diagonal Gell-Mann matrices.

    For completeness, j = k = 0 returns the identity matrix, which is not
    properly a Gell-Mann matrix.

    Args:
        j (int): operator index in `[0, dim - 1]`.
        k (int): operator index in `[0, dim - 1]`.
        dim (int): size of the output matrix.

    Returns:
        ndarray: Gell-Mann matrix `j, k` of dimension `dim`.
    """

    assert dim >= 2, "Dimension must be greater than 1."
    assert j >= 0 and k >= 0 and j < dim and k < dim, "Ill-formed indexes."

    if j == k:
        if j == 0:
            return np.eye(dim)
        else:
            gm = np.append(np.ones((j, 1)), -j)
            gm = np.append(gm, np.zeros((dim - len(gm))))
            return np.sqrt(2 / (j * (j + 1))) * np.diag(gm)
    else:
        if j < k:
            gm = np.zeros((dim ** 2, 1))
            gm[[j * dim + k, k * dim + j]] = 1
            return gm.reshape(dim, dim)
        else:
            gm = np.zeros((dim ** 2, 1), dtype=complex)
            gm[j * dim + k] = 1j
            gm[k * dim + j] = -1j
            return gm.reshape(dim, dim)


def gell_mann_matrices(dim=2):
    """All generalized Gell-Mann matrices for the given dimension.

    Args:
        dim (int): size of the output matrix.

    Returns:
        list (of ndarrays): identity + all `dim ** 2 - 1` Gell-Mann operators.
    """

    return [gell_mann_matrix(*jk, dim) for jk in product(range(dim), repeat=2)][1:]


def matrix2bloch(operator):
    """Convert an operator to a Bloch vector in the Gell-Mann matrices basis.

    Args:
        operator (ndarray): square matrix representation of an operator.

    Returns:
        list: coefficients for each Gell-Mann matrix in lexicographic ordering.

    Todo:
        * Further testing for dim > 2.
    """

    operator = np.asarray(operator)
    gms = gell_mann_matrices(operator.shape[0])
    return [np.trace(operator @ gm) for gm in gms]


def bloch2matrix(vec):
    """Convert a generalized Bloch vector to a matrix representation.

    .. math::
        \rho = \frac{\mathbb{1}_d}{d} + \frac{1}{2} \sum \lambda_i \mb{\sigma}_i
    Ref.: https://doi.org/10.1016/S0375-9601(03)00941-1

    Todo:
        * Check dim for perfect square.
        * Vectorize bloch_vec and gms multiplications.
    """

    dim = int(np.sqrt(len(vec) + 1))
    gms = gell_mann_matrices(dim)
    assert len(gms) == len(vec), "Dimension mismatch"

    gms = [vec[i] * gms[i] for i in range(len(vec))]
    return (1 / dim) * np.eye(dim) + sum(gms) / 2


"""Geometry and polyhedra"""

def insphere_radius(verts):
    """Radius of the biggest sphere inscribed in the convex hull of verts."""

    hull = ConvexHull(verts)
    return np.min(np.abs(hull.equations[:, -1]))  # abs as equations are Ax + b <= 0.




def rotate(vectors, alpha, beta, gamma):
    """3D rotation using Euler angles."""

    yaw = np.asarray([[cos(alpha), -sin(alpha), 0],
                      [sin(alpha), cos(alpha), 0],
                      [0, 0, 1]])
    pitch = np.asarray([[cos(beta), 0, sin(beta)],
                        [0, 1, 0],
                        [-sin(beta), 0, cos(beta)]])
    roll = np.asarray([[1, 0, 0],
                     [0, cos(gamma), -sin(gamma)],
                     [0, sin(gamma), cos(gamma)]])

    return (yaw @ pitch @ roll @ vectors.T).T


@normalize
def uniform_sphere(N, dim=3):
    """Uniform d-sphere sampling.

    Method: Sample from the normal distribution and normalize.
    Ref.: Sec. 2.1 @ http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    Args:
        N (int): nof. points to sample.
        dim (int): dimension where the sphere is embedded (nof. components in vector)

    Returns:
        ndarray: `N x dim` matrix where each row is a uniformly sampled point on the sphere.
    """

    return np.random.normal(0, 1, [N, dim])


def uniform_ball(N, dim=3):
    """Uniform d-ball sampling

    Method: Uniform sphere sampling in `dim + 2` then discard extra coordinates.
    Ref.: Sec. 3.1 @ http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    Args:
        N (int): nof. points to sample.
        dim (int): dimension where the ball is embedded (nof. components in vector)

    Returns:
        ndarray: `N x dim` matrix where each row is a uniformly sampled point in the ball.
    """

    coords = uniform_sphere(N, dim + 2)
    return coords[:, 0:dim]


@normalize
@antipodals
def octa():
    """Vertices of an octahedron."""

    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@normalize
@antipodals
def icos():
    """Vertices of an icosahedron."""

    a, b = 1, (1 + 5 ** 0.5) / 2

    return np.array([[0, a, b], [0, -a, b],
                     [a, b, 0], [-a, b, 0],
                     [b, 0, a], [-b, 0, a]])


@normalize
@antipodals
def romb():
    """Vertices of a rombicuboctahedron."""

    a, b = 1, 1 + 2 ** 0.5

    return np.array([[a, a, b], [-a, -a, b], [a, -a, b], [-a, a, b],
                      [a, b, a], [-a, -b, a], [a, -b, a], [-a, b, a],
                      [b, a, a], [-b, -a, a], [b, -a, a], [-b, a, a]])


@normalize
@antipodals
def dod():
    """Vertices of a dodecahedron."""

    a = (1 + 5 ** 0.5) / 2

    return np.array([[a, a, a], [-a, a, a], [a, -a, a], [-a, -a, a],
                     [0, a ** 2, 1], [0, -a ** 2, 1],
                     [a ** 2, 1, 0], [-a ** 2, 1, 0],
                     [1, 0, a ** 2], [-1, 0, a ** 2]])


@antipodals
def mirror_symmetric(theta):
    """Mirror-symmetric measurements vectors for given theta."""

    return np.asarray([(0, 0, 1),
                       (sin(theta), 0, cos(theta)),
                       (sin(-theta), 0, cos(-theta))])


"""Measurements utilities"""


def rand_projective_meas(N, dim=2):
    """Exactly N uniformly distributed projective measurements.

    For some draws of `uniform_sphere`, `hemispherectomy` may reduce `N`
    because of vertices in the equator. We fix `N` by brute force.

    WARNING: Only working for d = 2. Must generalize.

    Args:
        N (int): nof. points to sample.
        dim (int): state dimension (e.g.: 2 for qubit, 3 for qutrit etc.)

    Returns:
        ndarray: `N x (dim**2 - 1)` matrix where each row is a generalized Bloch
            vector of a projective effect in dimension `dim`.

    Todo:
        * Generalize for d > 2.
    """

    @antipodals
    @hemispherectomy
    def projective_nd(N, dim=2):
        """Approximately N uniformly distributed projective measurements."""

        return uniform_sphere(2 * N, dim)

    dim = dim ** 2 - 1  # Nof. components in a bloch vector in `dim`.
    while True:
        verts = projective_nd(N, dim)
        if verts.shape[0] == 2 * N: return verts
        else: continue


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


def plot_measurements(meas, insphere=True):
    """Plot measurement vertices on the Bloch sphere, their hull and the inscribed sphere."""

    chull = ConvexHull(meas)
    polys = Poly3DCollection([chull.points[simplex]
                              for simplex in chull.simplices])
    polys.set_edgecolor([.2, .2, .2, .4])
    polys.set_linewidth(.8)
    polys.set_facecolor('g')
    polys.set_alpha(.15)

    ax = Axes3D(plt.figure())
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_box_aspect([1,1,1])

    # Plot polytope
    ax.add_collection3d(polys)

    if insphere:
        # Plot insphere
        radius = insphere_radius(meas)
        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, np.pi, 50)
        x = radius * np.outer(np.cos(theta), np.sin(phi))
        y = radius * np.outer(np.sin(theta), np.sin(phi))
        z = radius * np.outer(np.ones(np.size(theta)), np.cos(phi))
        ax.plot_surface(x, y, z, shade=False, rstride=1,
                        cstride=1,alpha=.6, linewidth=0)
    return ax
