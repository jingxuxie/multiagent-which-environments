"""Check fixed-seed numerical endpoints against the audited first reference run."""
from pathlib import Path
import csv
import math
import json

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'finite_geometries': 120, 'approximation_sharpness': 6, 'dilution': 48,
            'structured_checks': 300, 'compatible_ambiguity': 16, 'interval_checks': 400,
            'calibration': 1200, 'curricula': 14, 'learned_seeds': 2016,
            'learning': 84, 'class_expansion': 100}


def verify():
    data = {}
    for name, count in EXPECTED.items():
        with (ROOT / 'results' / (name + '.csv')).open(newline='') as f:
            data[name] = list(csv.DictReader(f))
        assert len(data[name]) == count, (name, len(data[name]), count)
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    expected = json.loads((ROOT / 'docs/reference_summary.json').read_text())
    def walk(actual, wanted, path='summary'):
        if isinstance(wanted, dict):
            assert set(actual) == set(wanted), path
            for key in wanted: walk(actual[key], wanted[key], path + '.' + key)
        elif isinstance(wanted, (float, int)):
            assert math.isclose(float(actual), float(wanted), rel_tol=1e-7, abs_tol=1e-7), (path, actual, wanted)
        else:
            assert actual == wanted, (path, actual, wanted)
    walk(summary, expected)
    print('Reference row counts checked:', EXPECTED)
    print('All reference summary endpoints match within absolute/relative 1e-7')
    return data


if __name__ == '__main__':
    verify()
