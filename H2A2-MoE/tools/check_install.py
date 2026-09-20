#!/usr/bin/env python
"""Build all eight models and optionally validate released checkpoint files."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path,
                        help='Directory containing TASK.pth for all eight tasks.')
    args = parser.parse_args()
    import gc
    from importlib.metadata import version
    import torch
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet3d.registry import MODELS, METRICS
    from mmdet3d.utils import register_all_modules
    from mmengine.dataset import Compose
    from tools._eval_utils import require_evaluation_config

    print({p: version(p) for p in ('torch', 'mmengine', 'mmcv', 'mmdet',
                                   'mmdet3d', 'MinkowskiEngine')})
    register_all_modules(init_default_scope=True)
    torch.set_num_threads(2)
    for path in sorted((ROOT / 'configs').glob('*.py')):
        cfg = Config.fromfile(str(path))
        require_evaluation_config(cfg)
        Compose(cfg.test_pipeline)
        METRICS.build(cfg.test_evaluator)
        model = MODELS.build(cfg.model).cpu().eval()
        if args.checkpoints:
            checkpoint = args.checkpoints / (path.stem + '.pth')
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
        print('PASS', path.stem, 'parameters:', sum(p.numel() for p in model.parameters()))
        del model
        gc.collect()


if __name__ == '__main__':
    main()
