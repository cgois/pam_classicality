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
from more_itertools import random_permutation


ATOL = 1E-8


"""Generalities and operations"""

def chunks(lst, n):
    """Split lst into chunks of size n."""

    return [lst[i:i + n] for i in range(0, len(lst), max(1, n))]


def outer(vec1, vec2=None):
    """Outer product (with complex conjugation) between `vec1` and `vec2`

    If `vec2` is not supplied, return outer product of `vec1` with itself

    Args:
        vec1: ndarray either with shape (n,) or (n,1)
        vec2: ndarray either with shape (n,) or (n,1)

    Returns:
        ndarray: outer product (with complex conjugation) between `vec1` and
            `vec2`, or between `vec1` and itself it `vec2` not given.
"""

    if vec1.ndim == 1:
        vec1 = vec1[:,None]
    if vec2:
        if vec2.ndim == 1:
            vec2 = vec2[:,None]
    else:
        vec2 = vec1

    return vec1 @ vec2.conj().T


"""Booleans"""

def is_herm(matrix):
    """Check whether `matrix` is Hermitian."""

    return np.allclose(matrix, matrix.conj().T)


def is_psd(matrix):
    """Test `matrix` for positive semi-definiteness."""

    if is_herm(matrix):
        return np.all(np.linalg.eigvalsh(matrix) >= - ATOL)
    else:
        return False


def is_pd(matrix):
    """Test `matrix` for positive-definiteness."""

    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    else:
        return True


def is_projection(matrix):
    """Check whether `matrix` is a projection operator."""

    return np.allclose(matrix, matrix @ matrix)


def is_measurement(meas):
    """Check whether `meas` is a well-defined quantum measurement.

    Args:
        meas (list): list of ndarrays representing the measurement's effects.

    Returns:
        bool: returns `True` iff `meas` is composed of effects which are:
            - Square matrices with the same dimension,
            - Positive semi-definite, and
            - Sum to the identity operator.
    """

    dims = meas[0].shape

    try:
        square = len(dims) == 2 and dims[0] == dims[1]
        same_dim = np.all(np.asarray([eff.shape for eff in meas]) == dims)
        psd = np.all([is_psd(ef) for ef in meas])
        complete = np.allclose(sum(meas), np.eye(dims[0]))
    except (ValueError, np.linalg.LinAlgError):
        return False

    return square and same_dim and psd and complete


def is_projective_measurement(meas):
    """Check whether `meas` is a well-defined PVM.

    Args:
        meas (list): list of ndarrays representing the measurement's effects.

    Returns:
        bool: returns `True` iff `meas` is composed of effects which are:
            - Square matrices with the same dimension,
            - Positive semi-definite,
            - Sum to the identity operator and
            - Are projections.
    """

    return is_measurement and np.all([is_projection(e) for e in meas])


def is_dm(matrix):
    """Check whether `matrix` is a well-defined density matrix.

    Args:
        matrix (ndarray): density operator to test.

    Returns:
        bool: returns `True` iff `matrix` has unit-trace, is positive semi-definite
            and is hermitian.
    """

    try:
        trace_one = np.isclose(np.trace(matrix), 1)
        psd = is_psd(matrix)
        herm = is_herm(matrix)
    except (ValueError, np.linalg.LinAlgError):
        return False

    return trace_one and psd and herm


"""Decorators"""

def hemispherectomy(func):
    """Removes all 'southern hemisphere' vectors."""

    @functools.wraps(func)
    def hemispherectomy_wrapper(*args, **kwargs):
        vecs = func(*args, **kwargs)
        negs = [r for r in range(vecs.shape[0]) if vecs[r, -1] < 0]
        vecs = np.delete(vecs, negs, axis=0)

        return vecs
    return hemispherectomy_wrapper


def antipodals(func):
    """Takes a function that generates vectors and appends their antipodals

    The antipodals will be intercalated maintaining ordering.
    """

    @functools.wraps(func)
    def antipodals_wrapper(*args, **kwargs):
        vecs = func(*args, **kwargs)
        return np.ravel([vecs, -vecs], order="F").reshape(vecs.shape[1], -1).T
    return antipodals_wrapper


def normalize_rows(func):
    """Make an array generating function return it with normalized rows."""

    @functools.wraps(func)
    def normalize_wrapper(*args, **kwargs):
        array = func(*args, **kwargs)
        norm = np.sum(np.square(array), 1) ** 0.5

        return (array.T / norm).T
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


"""Operators and representations"""

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


def pauli_x():
    """X Pauli matrix."""

    return gell_mann_matrix(0, 1, dim=2)

def pauli_y():
    """Y Pauli matrix."""

    return gell_mann_matrix(1, 0, dim=2)

def pauli_z():
    """Z Pauli matrix."""

    return gell_mann_matrix(1, 1, dim=2)

def pauli():
    """List with Pauli matrices [X, Y, Z]."""

    return gell_mann_matrices(dim=2)


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
    return np.eye(dim) / dim + sum(gms) / 2


"""Random stuff"""

@normalize_rows
def uniform_sphere(N=1, dim=3):
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


def uniform_ball(N=1, dim=3):
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


def randnz(shape, norm=1/np.sqrt(2)):
    """Normally distributed complex number."""

    real = np.random.normal(0, 1, shape)
    imag = 1j * np.random.normal(0, 1, shape)
    return (real + imag) * norm


def random_unitary_haar(dim=2):
    """Random unitary matrix according to Haar measure.

    Ref.: https://arxiv.org/abs/math-ph/0609050v2
    """

    q, r = np.linalg.qr(randnz((dim, dim)))
    m = np.diagonal(r)
    m = m / np.abs(m)
    return np.multiply(q, m, q)


def random_unitary_bures(dim=2):
    pass


def random_pure_state(dim=2, density=True):
    """Generates a random pure quantum state of dimension `dim` in Haar measure.

    Takes first column of a Haar-random unitary operator.

    Args:
        dim: dimension of the state vectors (2 for qubits, 3 for qutrits etc.)
        density: if `True`, returns a density matrix instead of state vector.

    Returns:
        ndarray: a `dim`-length state vector if `density == False`, else a
            `dim x dim` density operator.
    """

    st = random_unitary_haar(dim)[:,0]
    if density:
        st = outer(st)
    return st


# TODO: Find a way to generate random projective measurements:
def random_projective_measurement(dim=2):
    """Generates a random projective measurement with rank-1 effects.

    A random unitary is sampled according to Haar measure. It's rows
    form a complete basis for the state space of dimension `dim`. We
    take the outer product of each row and return these projections.

    Args:
        dim: effects dimension.

    Returns:
        list: each element is a trace-1 PSD matrix, and elements sum to identity.
    """

    return [outer(eff) for eff in random_unitary_haar(dim)]


def random_povm(dim=2):
    pass



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


@normalize_rows
def tetrahedron():
    return np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])


@normalize_rows
@antipodals
def octa():
    """Vertices of an octahedron."""

    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@normalize_rows
@antipodals
def icos():
    """Vertices of an icosahedron."""

    a, b = 1, (1 + 5 ** 0.5) / 2

    return np.array([[0, a, b], [0, -a, b],
                     [a, b, 0], [-a, b, 0],
                     [b, 0, a], [-b, 0, a]])


@normalize_rows
@antipodals
def romb():
    """Vertices of a rombicuboctahedron."""

    a, b = 1, 1 + 2 ** 0.5

    return np.array([[a, a, b], [-a, -a, b], [a, -a, b], [-a, a, b],
                      [a, b, a], [-a, -b, a], [a, -b, a], [-a, b, a],
                      [b, a, a], [-b, -a, a], [b, -a, a], [-b, a, a]])


@normalize_rows
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
        * Generalize to measurements with different nof. effects.
    """

    if "solver" not in kwargs:
        kwargs["solver"] = "cvxopt"
    if "verb" not in kwargs:
        kwargs["verb"] = 0
    dim = len(measurements[0])  # Only works for measurements with dim effects

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
                                eta * meas[oper] + (1 - eta) * np.eye(dim) / dim)
        block_size *= dim

    prob.set_objective("max", eta)
    prob.options.solver = kwargs["solver"]
    prob.options.verbosity = kwargs["verb"]
    prob.license_warnings = False
    return prob.solve()


def plot_measurements(meas, insphere=True):
    """Plot measurement vertices on the Bloch sphere and their convex hull.

    Args:
        meas (ndarray): row-wise matrix of Bloch vectors.
        insphere (bool): plot largest sphere that can be inscribed in `meas`.

    Returns:
        matplotlib.Axes
    """

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
