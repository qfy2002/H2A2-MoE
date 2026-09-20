#!/usr/bin/env python
"""Audit the source release before uploading it (standard library only)."""
import argparse
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {'.git', '__pycache__', '.venv', '.pytest_cache',
                 'data', 'work_dirs', 'outputs', 'build', 'dist'}
FORBIDDEN_FILES = {
    'train.py', 'train_multi.py', 'dist_train.sh', 'dist_train_multitask.sh',
    'mtl_runner.py', 'ngh_optim_wrapper.py', 'task_finetune_hook.py',
    'task_config.py', 'task_config_finetune_task_params.py', 'manuscript.tex',
}
FORBIDDEN_SYMBOLS = {
    'MTL_Runner', 'TR3DNGHOptimWrapper', 'TR3DTaskOnlyOptimWrapper',
    'TR3DTaskSpecificFinetuneHook',
}
FORBIDDEN_SUFFIXES = {'.pth', '.pt', '.ckpt', '.pkl', '.bin', '.npy',
                      '.npz', '.so', '.obj', '.ply', '.zip', '.gz', '.log'}
CHECKPOINT_FILES = {
    'checkpoints/{}.pth'.format(task) for task in (
        's3dis_det', 'scannet_det', 'arkitscenes_det', 'multiscan_det',
        '3rscan_det', 'scannetpp_det', 'scannet_seg', 's3dis_seg')
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-local-checkpoints', action='store_true',
                        help='Allow the eight named weight assets locally; '
                             'they must remain outside the source upload.')
    args = parser.parse_args()
    errors = []
    count = 0
    local_weights = 0
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if set(relative.parts[:-1]) & EXCLUDED_DIRS:
            continue
        if path.is_symlink():
            errors.append(str(relative) + ': symlinks must not be packaged')
        if not path.is_file():
            continue
        if (args.allow_local_checkpoints and not path.is_symlink()
                and relative.as_posix() in CHECKPOINT_FILES):
            local_weights += 1
            continue
        count += 1
        if path.name in FORBIDDEN_FILES or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(str(relative) + ': excluded from the source release')
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(str(relative) + ': unexpected file larger than 10 MiB')
        if path.suffix == '.py':
            try:
                source = path.read_text()
                tree = ast.parse(source, filename=str(relative))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id in FORBIDDEN_SYMBOLS:
                        errors.append(str(relative) + ': imports training implementation')
                if str(relative).startswith('configs/'):
                    for node in tree.body:
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id in {
                                    'train_cfg', 'train_pipeline', 'train_dataloader',
                                    'optim_wrapper', 'param_scheduler', 'load_from', 'resume'}:
                                    errors.append(str(relative) + ': training/checkpoint setting')
                if path.name != 'check_release.py':
                    for prefix in ('/home/', '/hdd_data/', '/mnt/', '/tmp/'):
                        if prefix in source:
                            errors.append(str(relative) + ': private absolute path')
            except (SyntaxError, UnicodeError) as error:
                errors.append(str(error))
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    if len(list((ROOT / 'configs').glob('*.py'))) != 8:
        print('Expected eight task configurations.', file=sys.stderr)
        return 1
    print('PASS: {} source files; eight test configs; no excluded training files.'.format(count))
    if local_weights:
        print('{} local checkpoints excluded from the source audit.'.format(local_weights))
    return 0


if __name__ == '__main__':
    sys.exit(main())
