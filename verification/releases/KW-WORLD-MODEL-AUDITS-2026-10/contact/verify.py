#!/usr/bin/env python3
import argparse, json, unittest
from pathlib import Path
import numpy as np
import test_study
from make_report import render
ROOT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--replay');args=p.parse_args()
suite=unittest.defaultTestLoader.loadTestsFromModule(test_study)
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful(): raise SystemExit(1)
assert (ROOT/'REPORT.md').read_text()==render()
print('PASS report byte-for-byte generation')
if args.replay:
    a=json.loads((ROOT/'results.json').read_text());b=json.loads(Path(args.replay).read_text())
    for field in ['rows','summary','paired','selection','versions']:
        assert a[field]==b[field], f'Exact deterministic replay mismatch: {field}'
    print('PASS exact complete ten-seed replay; rows/summary/paired/selection/versions identical')
print('PASS verification complete')
