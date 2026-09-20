#!/usr/bin/env python
"""Run one TR3D detection or segmentation visualization job."""

import argparse
import copy
import os
import os.path as osp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mmengine.config import Config, DictAction

from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.utils import replace_ceph_backend
from tools._eval_utils import build_test_runner, set_data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Headless TR3D detection/segmentation OBJ export.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--data-root', help='Override the prepared dataset root.')
    parser.add_argument('--task', choices=['det', 'seg'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--score-thr', type=float, default=0.1)
    parser.add_argument('--max-scenes', type=int, default=10,
                        help='0 exports all scenes.')
    parser.add_argument(
        '--scene-index', type=int,
        help='Export only this zero-based test-dataset index.')
    parser.add_argument(
        '--scene-indices',
        help='Comma-separated zero-based test indices for scene ranking.')
    parser.add_argument(
        '--ranking-csv',
        help='Write per-scene segmentation metrics instead of OBJ files.')
    parser.add_argument('--work-dir')
    parser.add_argument('--ceph', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def limit_dataset(dataset_cfg, max_scenes):
    if 'dataset' in dataset_cfg:
        limit_dataset(dataset_cfg['dataset'], max_scenes)
    elif 'datasets' in dataset_cfg:
        if not dataset_cfg['datasets']:
            raise ValueError('Cannot limit an empty dataset list.')
        dataset_cfg['datasets'] = dataset_cfg['datasets'][:1]
        limit_dataset(dataset_cfg['datasets'][0], max_scenes)
    else:
        dataset_cfg['indices'] = max_scenes


def select_dataset_index(dataset_cfg, scene_index):
    if 'dataset' in dataset_cfg:
        select_dataset_index(dataset_cfg['dataset'], scene_index)
    elif 'datasets' in dataset_cfg:
        if not dataset_cfg['datasets']:
            raise ValueError('Cannot select from an empty dataset list.')
        dataset_cfg['datasets'] = dataset_cfg['datasets'][:1]
        select_dataset_index(dataset_cfg['datasets'][0], scene_index)
    else:
        dataset_cfg['indices'] = [scene_index]


def select_dataset_indices(dataset_cfg, scene_indices):
    if 'dataset' in dataset_cfg:
        select_dataset_indices(dataset_cfg['dataset'], scene_indices)
    elif 'datasets' in dataset_cfg:
        if not dataset_cfg['datasets']:
            raise ValueError('Cannot select from an empty dataset list.')
        dataset_cfg['datasets'] = dataset_cfg['datasets'][:1]
        select_dataset_indices(dataset_cfg['datasets'][0], scene_indices)
    else:
        dataset_cfg['indices'] = scene_indices


def prepare_seg_pipeline(cfg):
    pipeline = copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
    insert_at = len(pipeline)
    for index, transform in enumerate(pipeline):
        if transform.get('type') in ('TR3DOfficialSparseQuantize',
                                     'Pack3DDetInputs',
                                     'TR3DPack3DDetInputs'):
            insert_at = index
            break
    pipeline.insert(insert_at, dict(type='TR3DKeepPointsForVisualization'))

    meta_keys = list(Pack3DDetInputs.__init__.__defaults__[0])
    pack = pipeline[-1]
    if pack.get('type') not in ('Pack3DDetInputs', 'TR3DPack3DDetInputs'):
        raise TypeError('The segmentation test pipeline must end with a Pack transform.')
    if 'meta_keys' in pack:
        meta_keys = list(pack['meta_keys'])
    if 'visualization_points' not in meta_keys:
        meta_keys.append('visualization_points')
    pack['meta_keys'] = tuple(meta_keys)
    cfg.test_dataloader.dataset.pipeline = pipeline


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.ceph:
        cfg = replace_ceph_backend(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.data_root:
        set_data_root(cfg.test_dataloader.dataset, args.data_root)
    if args.max_scenes < 0:
        raise ValueError('--max-scenes must be >= 0.')
    if args.scene_index is not None and args.scene_index < 0:
        raise ValueError('--scene-index must be >= 0.')
    if args.scene_index is not None and args.scene_indices is not None:
        raise ValueError('Use only one of --scene-index and --scene-indices.')
    if args.ranking_csv is not None and args.task != 'seg':
        raise ValueError('--ranking-csv is only available for segmentation.')
    if args.scene_indices is not None:
        scene_indices = [
            int(value) for value in args.scene_indices.split(',') if value
        ]
        if not scene_indices or min(scene_indices) < 0:
            raise ValueError('--scene-indices must contain non-negative indices.')
        select_dataset_indices(cfg.test_dataloader.dataset, scene_indices)
    elif args.scene_index is not None:
        select_dataset_index(cfg.test_dataloader.dataset, args.scene_index)
    elif args.max_scenes > 0:
        limit_dataset(cfg.test_dataloader.dataset, args.max_scenes)
    if args.task == 'seg' and args.ranking_csv is None:
        prepare_seg_pipeline(cfg)

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.resume = False
    cfg.work_dir = args.work_dir or osp.join(
        args.output_dir, 'logs', args.dataset_name)
    if args.ranking_csv is not None:
        metric_cfg = dict(
            type='TR3DSegmentationRankingMetric',
            output_csv=args.ranking_csv)
    else:
        metric_type = ('TR3DDetectionVisMetric' if args.task == 'det'
                       else 'TR3DSegmentationVisMetric')
        metric_cfg = dict(
            type=metric_type,
            vis_dir=args.output_dir,
            dataset_name=args.dataset_name,
            max_samples=(1 if args.scene_index is not None else
                         None if args.max_scenes == 0 else args.max_scenes))
    if args.task == 'det':
        metric_cfg['score_thr'] = args.score_thr
    cfg.test_evaluator = metric_cfg

    runner = build_test_runner(cfg, args.checkpoint)
    runner.test()


if __name__ == '__main__':
    main()
