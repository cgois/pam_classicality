"""Measurements are given as Bloch vectors."""

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos

from decorators import antipodals, hemispherectomy, normalize


"""Utility functions:"""

def insphere_radius(verts):
    """Radius of the biggest sphere inscribed in the convex hull of verts."""

    hull = ConvexHull(verts)
    # hull.equations are Ax + b <= 0, hence the abs.
    return np.min(np.abs(hull.equations[:, -1]))


def bloch2density(bloch):
    """Takes Bloch vector for a qbit and returns its density operator."""
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return (1 / 2) * (np.eye(2) + bloch[0] * X + bloch[1] * Y + bloch[2] * Z)


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



"""Some useful polyhedra (rows are vectors):"""


@normalize
def uniform_sphere(N, dim=3):
    """Uniform d-sphere sampling. Each row is a vector.

    Method: Sample from the normal distribution and normalize.
    Ref.: Sec. 2.1 @ http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """

    return np.random.normal(0, 1, [N, dim])


def uniform_ball(N, dim=3):
    """Uniform d-ball sampling. Each row is a vector.

    Method: Sample from the normal distribution and normalize.
    Ref.: Sec. 3.1 @ http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """

    coords = uniform_sphere(N, dim + 2)
    return coords[:, 0:dim]


def projective(N, dim=3):
    """Returns exactly N uniformly distributed projective measurements."""

    @antipodals
    @hemispherectomy
    def projective_nd(N, dim=3):
        """Returns approximately N uniformly distributed projective measurements."""

        return uniform_sphere(2 * N, dim)

    # NOTE: For some draws of uniform_sphere, hemispherectomy may reduce N (vecs. around equator)
    # We fix N by brute force, which is a bit stupid but is fast anyways.
    while True:
        verts = projective_nd(N)
        if verts.shape[0] == 2 * N: return verts
        else: continue


@normalize
@antipodals
def octa():
    """Returns the vertices of an octahedron."""

    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@normalize
@antipodals
def icos():
    """Returns the vertices of an icosahedron."""

    a, b = 1, (1 + 5 ** 0.5) / 2

    return np.array([[0, a, b], [0, -a, b],
                     [a, b, 0], [-a, b, 0],
                     [b, 0, a], [-b, 0, a]])


@normalize
@antipodals
def romb():
    """Returns the vertices of a rombicuboctahedron."""

    a, b = 1, 1 + 2 ** 0.5

    return np.array([[a, a, b], [-a, -a, b], [a, -a, b], [-a, a, b],
                      [a, b, a], [-a, -b, a], [a, -b, a], [-a, b, a],
                      [b, a, a], [-b, -a, a], [b, -a, a], [-b, a, a]])


@normalize
@antipodals
def dod():
    """Returns the vertices of a dodecahedron."""

    a = (1 + 5 ** 0.5) / 2

    return np.array([[a, a, a], [-a, a, a], [a, -a, a], [-a, -a, a],
                     [0, a ** 2, 1], [0, -a ** 2, 1],
                     [a ** 2, 1, 0], [-a ** 2, 1, 0],
                     [1, 0, a ** 2], [-1, 0, a ** 2]])


@antipodals
def mirror_symmetric(theta):
    """Return all our mirror-symmetric measurements for the given theta."""

    return np.asarray([(0, 0, 1),
                       (sin(theta), 0, cos(theta)),
                       (sin(-theta), 0, cos(-theta))])
