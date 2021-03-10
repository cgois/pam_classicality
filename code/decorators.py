import functools
import time
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tqdm


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
