"""Compare every scientific CSV cell, permitting numeric cross-platform roundoff."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path


def compare(left: Path, right: Path, atol: float = 1e-8) -> dict:
    names = sorted(p.name for p in left.glob('*.csv'))
    if names != sorted(p.name for p in right.glob('*.csv')):
        raise AssertionError('Scientific CSV inventories differ')
    cells = 0
    exact = 0
    for name in names:
        a, b = left / name, right / name
        exact += a.read_bytes() == b.read_bytes()
        with a.open(newline='') as fa, b.open(newline='') as fb:
            ra, rb = list(csv.reader(fa)), list(csv.reader(fb))
        if len(ra) != len(rb): raise AssertionError(f'{name}: different row counts')
        for row, (xa, xb) in enumerate(zip(ra, rb)):
            if len(xa) != len(xb): raise AssertionError(f'{name}:{row}: different widths')
            for col, (va, vb) in enumerate(zip(xa, xb)):
                cells += 1
                if va == vb: continue
                try: na, nb = float(va), float(vb)
                except ValueError:
                    raise AssertionError(f'{name}:{row}:{col}: text differs: {va!r}, {vb!r}')
                if math.isnan(na) and math.isnan(nb): continue
                if not math.isclose(na, nb, rel_tol=atol, abs_tol=atol):
                    raise AssertionError(f'{name}:{row}:{col}: {na} != {nb}')
    return dict(csv_files=len(names), byte_identical_files=exact, checked_cells=cells,
                absolute_relative_tolerance=atol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path); parser.add_argument('right', type=Path)
    parser.add_argument('--atol', type=float, default=1e-8)
    args = parser.parse_args()
    print(compare(args.left, args.right, args.atol))
