"""Run benchmark stages from the repository root."""
import argparse
import subprocess
import sys
from pathlib import Path

STAGES = {
    'standardize': '2b_standardize_system_ideas.py',
    'alignment': '3b_alignment_scoring.py',
    'evaluate': '5_axis_evaluation.py',
    'summary': '6_three_axis_summary.py',
    'native-resource-check': 'run_native_resource_check.py',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['list'] + list(STAGES))
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if args.stage == 'list':
        for name, script in STAGES.items():
            print(f'{name}: {script}')
        return
    root = Path(__file__).resolve().parents[1]
    target = root / 'benchmark/scripts' / STAGES[args.stage]
    if not target.is_file():
        raise FileNotFoundError(f'Benchmark stage is missing: {target}')
    if extra[:1] == ['--']:
        extra = extra[1:]
    command = [sys.executable, str(target), *extra]
    if args.dry_run:
        print('cwd:', root)
        print('command:', repr(command))
        return
    raise SystemExit(subprocess.call(command, cwd=root))

if __name__ == '__main__':
    main()
