#!/usr/bin/env python3

from pathlib import Path
from subprocess import check_call

TESTS_ROOT = Path(__file__).resolve(strict=True).parent
TEST_DATA_ROOT = TESTS_ROOT / 'data'
INPUTS_ROOT = TEST_DATA_ROOT / 'inputs'
VALIDATION_ROOT = TEST_DATA_ROOT / 'validation'
BROADPHASE_ROOT = TESTS_ROOT.parent
UTILS_ROOT = BROADPHASE_ROOT / 'utils'

SEED = 0
DENSITY = (1, 1000)
SIZE_RANGE = (1, 10)
COUNT = 10_000

TEST_DATA_ROOT.mkdir(mode=0o755, exist_ok=True)
INPUTS_ROOT   .mkdir(mode=0o755, exist_ok=True)

def main():
    in_name = (f'boxes-seed_{SEED:d}'
        f'-d_{DENSITY[0]:d}_{DENSITY[1]:d}'
        f'-s_{SIZE_RANGE[0]:d}_{SIZE_RANGE[1]:d}'
        f'-n_{COUNT:06d}.br_scene')
    in_path = INPUTS_ROOT / f'{in_name!s}'
    out_path = VALIDATION_ROOT
    args = ('gen_validation_data',
        '--in', f'{in_path!s}',
        '--out', f'{out_path!s}')
    cmd = ('cargo', 'run', '--bin', 'gen_test_data', '--', *args)
    print(' '.join(cmd))
    check_call(cmd, cwd=UTILS_ROOT)

main()