#!/usr/bin/env python3

from pathlib import Path
from subprocess import check_call

TESTS_ROOT = Path(__file__).resolve(strict=True).parent
TEST_DATA_ROOT = TESTS_ROOT / 'data'
INPUTS_ROOT = TEST_DATA_ROOT / 'inputs'
BROADPHASE_ROOT = TESTS_ROOT.parent
UTILS_ROOT = BROADPHASE_ROOT / 'utils'

SEED = 0
DENSITY = (1, 1000)
SIZE_RANGE = (1, 10)

TEST_DATA_ROOT.mkdir(mode=0o755, exist_ok=True)
INPUTS_ROOT   .mkdir(mode=0o755, exist_ok=True)

for count in (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000):
    name = (f'boxes-seed_{SEED:d}'
        f'-d_{DENSITY[0]:d}_{DENSITY[1]:d}'
        f'-s_{SIZE_RANGE[0]:d}_{SIZE_RANGE[1]:d}'
        f'-n_{count:06d}.br_scene')
    path = INPUTS_ROOT / f'{name!s}'
    args = ('gen_boxes',
        '--seed', f'{SEED:d}',
        '--count', f'{count:d}',
        '--density', f'{(DENSITY[0] / DENSITY[1]):f}',
        '--size_range', f'{SIZE_RANGE[0]:d}', f'{SIZE_RANGE[1]:d}',
        '--out', f'{path!s}')
    cmd = ('cargo', 'run', '--bin', 'gen_test_data', '--', *args)
    print(' '.join(cmd))
    check_call(cmd, cwd=UTILS_ROOT)