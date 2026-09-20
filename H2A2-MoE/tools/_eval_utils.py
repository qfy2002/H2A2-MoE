"""Utilities shared by evaluation and OBJ export."""
from pathlib import Path


def set_data_root(dataset, root):
    if 'datasets' in dataset:
        for child in dataset['datasets']:
            set_data_root(child, root)
    elif 'dataset' in dataset:
        set_data_root(dataset['dataset'], root)
    else:
        dataset['data_root'] = str(Path(root).expanduser().resolve())


def require_evaluation_config(cfg):
    for key in ('train_dataloader', 'train_cfg', 'optim_wrapper',
                'param_scheduler', 'runner_type', 'custom_hooks'):
        if cfg.get(key):
            raise ValueError('Use an evaluation-only config; found ' + key)
    if cfg.test_dataloader.get('batch_size', 1) != 1:
        raise ValueError('Keep batch_size=1: routing depends on the local batch.')
    cfg.resume = False


def build_test_runner(cfg, checkpoint):
    from mmengine.runner import Runner

    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError('Checkpoint not found: ' + str(checkpoint))
    require_evaluation_config(cfg)
    # Use strict loading to prevent evaluating partially loaded models.
    cfg.load_from = None
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(str(checkpoint), map_location='cpu', strict=True)
    return runner
