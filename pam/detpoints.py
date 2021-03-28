"""
Prepare-and-measure scenario deterministic strategies generation.

Usage: call "python detpoints.py --help"
"""

import argparse
from itertools import product, chain, permutations
from random import sample, randrange, randint

import numpy as np
from scipy.sparse import coo_matrix
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
        except OverflowError:  # Can't use unique sampling if n_lambdas is too large.
            samples = [randint(0, n_lambdas) for _ in range(samples)]
        except ValueError:  # If more than n_lambdas points requested, use n_lambdas.
            samples = range(0, n_lambdas)
        detp = [np.base_repr(el + shift, base=mb)[-width:] for el in samples]
    else:  # All strategies.
        detp = [np.base_repr(el + shift, base=mb)[-width:] for el in range(n_lambdas)]

    detp = [[int(digit) for digit in el] for el in detp]

    if segmented:
        return [[detp[i][j:j+my] for j in range(0, width, my)] for i in range(len(detp))]
    else:
        return detp


def build_detpoints(ma, mb, mx, my, samples=0, binary=True, normalized=False):
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
        normalized: if `True`, will remove each `mb`-th outcome from binary format.

    Returns:
        list: each row is a strategy in which the coefficients `i` stand for the
            deterministic result represented in the given strategy. Example: with
            `ma = 2`, `mb = 3`, `mx = 3`, `my = 2`, a row `[0, 2, 1, 0, 0, 2]`
            is the strategy where measurements y in preparations x return the i-th
            result. A conventional ordering is to take the lexicographic one:
            `[(x0, y0), (x0, y1), (x1, y0), (x1, y1), (x2, y0), (x2, y1)]`
    """

    indeps = np.asarray(independent_strategies(ma, mb, my, samples, segmented=True))

    # Find `n_ords` random orderings for the independent segments.
    if samples:
        # WARNING: If not enough independent samples, will not return *exactly* `samples`:
        n_ords = samples // len(indeps)
        orderings = []
        while len(orderings) != n_ords:
            order = random_product(range(ma), repeat=mx)
            if iselement(range(ma), order):
                orderings.append(order)
    # Find *all* allowed oderings for the segments (thus generating *all* det. points).
    else:
        orderings = [r for r in product(range(ma), repeat=mx) if iselement(range(ma), r)]

    detps = np.unique(indeps[:,orderings,:].reshape(-1, mx * my), axis=0)
    if binary:
        # Write down the position of each "1" as a sparse matrix then get dense one:
        b_0s = range(0, mb * mx * my, mb)
        idxs = detps + b_0s  # Position to indicate each outcome.
        r, c = idxs.shape
        row = np.concatenate([np.ones(c, dtype=int) * i for i in range(r)])
        col = np.concatenate(idxs)
        detps = np.asarray(coo_matrix((np.ones_like(col), (row, col))).todense())

        if normalized:
            detps = np.delete(detps, b_0s, axis=1)

    return detps


def symmetries(mb, mx, my):
    """List all symmetries in the behaviors."""

    names = [[[f"p({b}|{x}{y})" for b in range(mb)] for y in range(my)] for x in range(mx)]
    names = np.asarray(names)

    prep_perms = names[list(permutations(range(mx))),:,:]
    prep_perms = [np.concatenate(prep_perms[i,:,:,:], axis=None)
                  for i in range(prep_perms.shape[0])]

    meas_perms = names[:,list(permutations(range(my))),:]
    meas_perms = [np.concatenate(meas_perms[:,i,:,:], axis=None)
                  for i in range(meas_perms.shape[1])]

    out_perms = names[:,:,list(permutations(range(mb)))]
    out_perms = [np.concatenate(out_perms[:,:,i,:], axis=None)
                 for i in range(out_perms.shape[2])]

    maps = out_perms + prep_perms + meas_perms
    maps = "\n".join([" ".join(m) for m in maps])
    names = " ".join(names.reshape(mb * mx * my))

    return names, maps


def export(mb, mx, my, detps, fname, named=True, symms=True, lrs=False):
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
            names, maps = symmetries(mb - 1, mx, my)  # -1 to account for normalization.
            exp = "Names\n" + names + "\n"
            if symms:
                exp += "Maps\n" + maps + "\n"
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

    detps = build_detpoints(args.ma, args.mb, args.mx, args.my,
                            binary=True, normalized=True)
    export(args.mb, args.mx, args.my, detps, args.output,
           args.nonames, args.nosymms, args.lrs)
