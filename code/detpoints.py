"""
Usage: call "python detpoints.py --help"

Only works for mb = 2
Each element of each strategy is of the form p(b = 0 | a, y)

Todo:
    * Extend for all `mb`.
"""

import argparse
from itertools import product, chain, permutations
import numpy as np


def iselement(els, l):
    """Certify all elements in els are in l."""

    for el in els:
        if el not in l:
            return False
    return True


def independent_strategies(ma, mb, my):
    """Generate all single party strategies of length ma * my

    Enumerate n_lambdas integers and convert to a binary string with a suitable
    fixed length, then convert to list of integers.

    Args:
        ma: Nof. independent preparations.
        mb: Nof. possible measurement results (nof. effects)
        my: Nof. allowed measurements.
    """

    cpp = my * (mb - 1)  # nof. coefficients per preparation.
    stlen = ma *  cpp  # length of each strategy.
    n_lambdas = ma ** stlen
    detp = ["{:0{}b}".format(el, stlen) for el in range(n_lambdas)]
    detp = [[int(digit) for digit in el] for el in detp] # Convert binary str to list of ints.

    # Split a's in each strategy, e.g.: [0, 1, 1, 0] -> [[0, 1], [1, 0]] if ma = my = 2.
    return [[detp[i][j:j+cpp] for j in range(0, stlen, cpp)] for i in range(n_lambdas)]


def detpoints(ma, mb, mx, my):
    """Build a PAM scenario deterministic strategies.

    Sample ma independent strategies and repeat mx - ma blocks in allowed
    positions. Each row is a deterministic strategy. Coefficients stand for
    [p(0 | 0, 0), ..., p(mb-1 | 0, 0), p(0 | 0, 1), ..., p(mb-1 | mx my)]
    """

    indeps = independent_strategies(ma, mb, my)

    # All possible message orderings: mx times [0, ..., ma] x ... x [0, ..., ma]
    # but take only the ones with all independent ma's in it.
    redundants = [r for r in product(range(ma), repeat=mx) if iselement(range(ma), r)]

    detps = []
    for i in range(len(indeps)):
        for messages in redundants:
            detps.append([indeps[i][a] for a in messages])

    # Flatten the sublists and return unique strategies.
    detps = [list(chain(*detps[i])) for i in range(len(detps))]
    return [el for idx,el in enumerate(detps) if el not in detps[:idx]]


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
