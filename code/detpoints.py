"""
Usage: call "python detpoints.py --help"
"""

import argparse
from itertools import product, chain, permutations
from random import sample, randrange, randint

import numpy as np


def iselement(els, l):
    """Certify all elements in els are in l."""

    for el in els:
        if el not in l:
            return False
    return True


def independent_strategies(ma, mb, my, samples=0, segmented=False):
    """Generate all single party strategies of length ma * my

    Enumerate n_lambdas integers and convert to an `mb`-ary representation string,
    then convert to a list of integers.

    Args:
        ma: nof. independent preparations.
        mb: nof. effects per measurement.
        my: nof. available measurements.
        samples: if `0`, generates all strategies, else sample `samples` strategies.
        segmented: segment coeffients corresponding to each preparation.
            e.g.: if ma = my = 2, then [0, 1, 1, 0] -> [[0, 1], [1, 0]].

    Returns:
        list: each row is a strategy in which the coefficients `i` stand for the
            deterministic result represented in the given strategy. Example: with
            `ma = 2`, `mb = 3`, `my = 2`, a row `[[0, 2], [1, 0]]` is the strategy
            where the operations `[[(a0, y0), (a0, y1)], [(a1, y0), (a1, y1)]]`
            return results `0, 2, 1` and `0`, respectively.
    """

    width = ma * my
    shift = mb ** width  # To include trailing zeros.
    n_lambdas = mb ** width

    if samples:
        try:
            samples = sample(range(0, n_lambdas), samples)
        except OverflowError: # Can't use unique sampling if n_lambdas is too large.
            samples = [randint(0, n_lambdas) for _ in range(samples)]
        detp = [np.base_repr(el + shift, base=mb)[-width:] for el in samples]
    else:  # All strategies.
        detp = [np.base_repr(el + shift, base=mb)[-width:] for el in range(n_lambdas)]

    detp = [[int(digit) for digit in el] for el in detp]

    if segmented:
        return [[detp[i][j:j+my] for j in range(0, width, my)] for i in range(n_lambdas)]
    else:
        return detp


def detpoints(ma, mb, mx, my, samples=0):
    """Build a PAM scenario's deterministic strategies.

    Sample independent strategies for `ma` preparations then repeat `mx - ma` blocks
    in all meaningful positions. See ../docs/detpoints.tex.

    Args:
        ma: nof. independent preparations.
        mb: nof. effects per measurement.
        mx: nof. total preparations.
        my: nof. available measurements.
        samples: if `0`, generates all strategies, else sample `samples` strategies.

    Returns:
        list: each row is a strategy in which the coefficients `i` stand for the
            deterministic result represented in the given strategy. Example: with
            `ma = 2`, `mb = 3`, `mx = 3`, `my = 2`, a row `[[0, 2], [1, 0], [0, 2]]`
            is the strategy where measurements y in preparations x return the i-th
            result. A conventional ordering is to take the lexicographic one:
            `[[(x0, y0), (x0, y1)], [(x1, y0), (x1, y1)], [(x2, y0), (x2, y1)]]`

    Todo:
        * Sampling version (use more_itertools.random_product)
    """

    indeps = np.asarray(independent_strategies(ma, mb, my, samples, segmented=True))

    # Indexes to include needed redundant segments in all allowed positions.
    orderings = [r for r in product(range(ma), repeat=mx) if iselement(range(ma), r)]
    detps = indeps[:,orderings,:].reshape(-1, mx * my)

    return np.unique(detps, axis=0)


def symmetries(mb, mx, my):
    """List all symmetries in the behaviors.

    Todo:
        * Clearer procedure for measurement permutation symmetries.
        * Include combined symmetries (measurement + preparation).
        * Extend for mb > 2.
    """

    names = [[f"p(0|{x}{y})" for y in range(1, my + 1)] for x in range(1, mx + 1)]

    # A very bad way of doing it:
    meas_perms = []
    for perm in permutations(range(my), my):
        equiv = [np.asarray(names)[prep][[perm]] for prep in range(mx)]
        meas_perms.append(np.asarray(equiv).reshape(1, -1))
    meas_perms = np.asarray(meas_perms).reshape(np.math.factorial(my), -1)
    meas_perms = "\n".join([" ".join(p) for p in meas_perms])

    prep_perms = list(permutations(names, mx))
    prep_perms = np.asarray(prep_perms).reshape(np.math.factorial(mx), mx * my)
    prep_perms = "\n".join([" ".join(p) for p in prep_perms])

    names = " ".join(np.asarray(names).reshape(mx * my))

    return "Names\n" + names + "\nMaps\n" + meas_perms + "\n" + prep_perms + "\n"


def export(detps, fname, named, lrs, mx, my):
    """Export detps to an lrs or PANDA-readable file, overwriting fname if it exists.

    Each row is a vertex p(b = 0 | a, y).

    First line is the file header, and the second adds a "1" before each vertex
    to indicate it's a vertex and not a ray, as specified in lrs's user guide.

    TODO:
    - Write the 'maps' section in PANDA output to filter redundant inequalities.
    """

    if lrs:
        exp = f"V-representation\nbegin\n{len(detps)} {len(detps[0]) + 1} rational\n"
        exp += "\n".join([" ".join(map(str, [1, *strategy])) for strategy in detps])
    else:
        if named:
            exp = symmetries(mb, mx, my)
        else:
            exp = ""
        exp += f"Vertices\n" + "\n".join([" ".join(map(str, strategy)) for strategy in detps])

    with open(fname, 'w') as file:
        file.write(exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ma', type=int, help='Dimension of classical message.')
    parser.add_argument('mb', type=int, help='Nof. measurement effects.')
    parser.add_argument('mx', type=int, help='Nof. preparations.')
    parser.add_argument('my', type=int, help='Nof. measurements')
    parser.add_argument('output', type=str, help='Output file name')
    parser.add_argument('--named', action='store_true', help='Named variables')
    parser.add_argument('--lrs', action='store_true', help='Output in lrs format')
    args = parser.parse_args()

    export(detstrategies(args.ma, 2, args.mx, args.my),
           args.output, args.named, args.lrs, args.mb, args.mx, args.my)
