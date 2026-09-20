#!/usr/bin/env python
"""Evaluate one H2A2-MoE dataset-task checkpoint with upstream MMEngine."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from mmengine.config import Config, DictAction
    from tools._eval_utils import build_test_runner, set_data_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--data-root', help='Override the prepared dataset root.')
    parser.add_argument('--work-dir')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Fix the random seed; omitted by default for a fresh seed per run.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.data_root:
        set_data_root(cfg.test_dataloader.dataset, args.data_root)
    cfg.launcher = args.launcher
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    cfg.work_dir = args.work_dir or str(Path('work_dirs') / Path(args.config).stem)
    runner = build_test_runner(cfg, args.checkpoint)
    runner.test()


if __name__ == '__main__':
    main()
