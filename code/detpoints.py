"""
Prepare-and-measure scenario deterministic strategies generation.

Usage: call "python detpoints.py --help"
"""

import argparse
from itertools import product, chain, permutations
from random import sample, randrange, randint

import numpy as np
from more_itertools import random_product


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
        return [[detp[i][j:j+my] for j in range(0, width, my)] for i in range(len(detp))]
    else:
        return detp


def detpoints(ma, mb, mx, my, samples=0, binary=False):
    """Build a PAM scenario's deterministic strategies.

    Sample independent strategies for `ma` preparations then repeat `mx - ma` blocks
    in all meaningful positions. See ../docs/detpoints.tex.

    Args:
        ma: nof. independent preparations.
        mb: nof. effects per measurement.
        mx: nof. total preparations.
        my: nof. available measurements.
        samples: if `0`, generates all strategies, else sample `samples` strategies.
        binary: if `True`, will expand the `mb`-ary basis in `mb` coefficients with
            a single `1` indicating the obtained outcome. E.g.: will turn a strategy
            [0, 2, 1, 0] into [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], which is the more
            conventional way of listing strategies.

    Returns:
        list: each row is a strategy in which the coefficients `i` stand for the
            deterministic result represented in the given strategy. Example: with
            `ma = 2`, `mb = 3`, `mx = 3`, `my = 2`, a row `[0, 2, 1, 0, 0, 2]`
            is the strategy where measurements y in preparations x return the i-th
            result. A conventional ordering is to take the lexicographic one:
            `[(x0, y0), (x0, y1), (x1, y0), (x1, y1), (x2, y0), (x2, y1)]`

    Todo:
        * binary version
    """

    indeps = np.asarray(independent_strategies(ma, mb, my, samples, segmented=True))

    if samples:
        # Find one allowed random ordering for the segments.
        orderings = random_product(range(ma), repeat=mx)
        while not iselement(range(ma), orderings):
            orderings = random_product(range(ma), repeat=mx)
    else:
        # Find *all* allowed oderings for the segments (thus generating *all* det. points).
        orderings = [r for r in product(range(ma), repeat=mx) if iselement(range(ma), r)]

    detps = indeps[:,orderings,:].reshape(-1, mx * my)
    return np.unique(detps, axis=0)


def symmetries(mb, mx, my):
    """List all symmetries in the behaviors.

    Todo:
        * Testing
    """

    names = [[[f"p({b}|{x}{y})" for b in range(mb)] for y in range(my)] for x in range(mx)]

    outcome_perms = list(permutations(range(mb)))
    prep_perms = list(permutations(range(mx)))
    meas_perms = list(permutations(range(my)))

    maps = np.asarray(names)[:,:,outcome_perms]
    maps = np.asarray([maps[prep_perms,:,:,i] for i in range(mb)])
    maps = np.asarray([maps[i,j,:,meas_perms,:]
                       for i in range(maps.shape[0])
                       for j in range(maps.shape[1])])


    maps = "\n".join([" ".join(m) for m in maps.reshape(-1, mb * mx * my)])
    names = " ".join(np.asarray(names).reshape(mb * mx * my))

    return names, maps


def export(mb, mx, my, detps, fname, named=True, symmetries=True, lrs=False):
    """Export `detps` to an lrs or PANDA-readable file named `fname` (will overwrite).

    Vertices of `detps` should be given row-wise. If `lrs`, the output will have
    `1` before each vertex to indicate it's a vertex and not a ray, as specified
    in lrs's user guide.

    Args:
        mb: nof. effects per measurement.
        mx: nof. total preparations.
        my: nof. available measurements.
        detps: row-wise listing of deterministic strategies to format.
        fname: output file path (will be overwriten).
        named: if `True`, will add PANDA's "Names" section to ease of reading.
        symmetries: if `True`, will add PANDA's "Maps" so that its convex hull
            algorithm will list a single representant for each inequality class.
        lrs: if `True`, will instead output for lrs (no names and symmetries support).
    """

    if lrs:
        exp = f"V-representation\nbegin\n{len(detps)} {len(detps[0]) + 1} rational\n"
        exp += "\n".join([" ".join(map(str, [1, *strategy])) for strategy in detps])
    else:
        if named:
            names, maps = symmetries(mb, mx, my)
            exp = "Names\n" + names + "\nMaps\n" + maps + "\n"
        else:
            exp = ""
        exp += f"Vertices\n" + "\n".join([" ".join(map(str, strategy)) for strategy in detps])

    with open(fname, 'w') as file:
        file.write(exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ma", type=int, help="Dimension of classical message.")
    parser.add_argument("mb", type=int, help="Nof. measurement effects.")
    parser.add_argument("mx", type=int, help="Nof. preparations.")
    parser.add_argument("my", type=int, help="Nof. measurements")
    parser.add_argument("output", type=str, help="Output file name")
    parser.add_argument("--nonames", action="store_false", help="No variables names")
    parser.add_argument("--nosymms", action="store_false", help="Don't list symmetries")
    parser.add_argument("--lrs", action="store_true", help="Output in lrs format")
    args = parser.parse_args()

    detps = detstrategies(args.ma, args.mb, args.mx, args.my)
    export(args.mb, args.mx, args.my, detps, args.fname, args.nonames, args.nosymms, args.lrs)
