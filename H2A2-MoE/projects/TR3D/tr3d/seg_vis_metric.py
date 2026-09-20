"""Headless export and scene ranking for semantic segmentation."""

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


S3DIS_PALETTE = (
    (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0),
    (255, 0, 255), (100, 100, 255), (200, 200, 100), (170, 120, 200),
    (255, 0, 0), (200, 100, 100), (10, 200, 100), (200, 200, 200),
    (50, 50, 50),
)

SCANNET_PALETTE = (
    (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120),
    (188, 189, 34), (140, 86, 75), (255, 152, 150), (214, 39, 40),
    (197, 176, 213), (148, 103, 189), (196, 156, 148), (23, 190, 207),
    (247, 182, 210), (219, 219, 141), (255, 127, 14), (158, 218, 229),
    (44, 160, 44), (112, 128, 144), (227, 119, 194), (82, 84, 163),
)


def _seg_palette(num_classes: int, dataset_name: str) -> np.ndarray:
    dataset_name = dataset_name.lower()
    if 's3dis' in dataset_name and num_classes == len(S3DIS_PALETTE):
        colors = list(S3DIS_PALETTE)
    elif 'scannet' in dataset_name and num_classes == len(SCANNET_PALETTE):
        colors = list(SCANNET_PALETTE)
    else:
        colors = []
        for label in range(num_classes):
            colors.append(((label * 37 + 53) % 256,
                           (label * 97 + 89) % 256,
                           (label * 173 + 127) % 256))
    colors.append((80, 80, 80))
    return np.asarray(colors, dtype=np.uint8)


def _write_seg_obj(points: np.ndarray, labels: np.ndarray, path: Path,
                   num_classes: int, ignore_index: Optional[int],
                   dataset_name: str) -> None:
    palette = _seg_palette(num_classes, dataset_name)
    labels = labels.astype(np.int64, copy=False)
    with path.open('w') as file:
        for point, label in zip(points, labels):
            if ignore_index is not None and int(label) == ignore_index:
                color = palette[-1]
            else:
                color = palette[int(label) % num_classes]
            file.write(
                f'v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} '
                f'{int(color[0])} {int(color[1])} {int(color[2])}\n')


@METRICS.register_module()
class TR3DSegmentationVisMetric(SegMetric):
    """Evaluate segmentation and export colored points as OBJ files."""

    def __init__(self,
                 vis_dir: str,
                 dataset_name: str = 'unknown',
                 max_samples: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.vis_dir = Path(vis_dir)
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self._exported_samples = 0

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch, data_samples)
        for data_sample in data_samples:
            if self.max_samples is not None and self._exported_samples >= self.max_samples:
                break
            if hasattr(data_sample, 'metainfo'):
                points = data_sample.metainfo.get(
                    'visualization_points', None)
            else:
                metainfo = data_sample.get('metainfo', {})
                points = data_sample.get('visualization_points',
                                         metainfo.get('visualization_points'))
            if points is None:
                raise KeyError(
                    'visualization_points is missing. The visualization '
                    'test pipeline was not prepared correctly.')
            self._export_scene(_numpy(points), data_sample)
            self._exported_samples += 1

    def _export_scene(self, points: np.ndarray, data_sample: dict) -> None:
        eval_ann = data_sample['eval_ann_info']
        gt_labels = _numpy(eval_ann['pts_semantic_mask'])
        pred_labels = _numpy(
            data_sample['pred_pts_seg']['pts_semantic_mask'])
        if len(points) != len(gt_labels) or len(points) != len(pred_labels):
            raise ValueError(
                f'Point/mask length mismatch for {self._scene_name(data_sample)}: '
                f'points={len(points)}, gt={len(gt_labels)}, pred={len(pred_labels)}')

        scene_name = self._scene_name(data_sample)
        scene_dir = self.vis_dir / self.dataset_name / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        num_classes = len(self.dataset_meta['classes'])
        ignore_index = self.dataset_meta.get('ignore_index', None)
        _write_seg_obj(points, gt_labels,
                       scene_dir / f'{scene_name}_gt.obj', num_classes,
                       ignore_index, self.dataset_name)
        _write_seg_obj(points, pred_labels,
                       scene_dir / f'{scene_name}_pred.obj', num_classes,
                       ignore_index, self.dataset_name)

        palette = _seg_palette(num_classes, self.dataset_name)
        with (scene_dir / 'class_palette.txt').open('w') as file:
            for label, class_name in enumerate(self.dataset_meta['classes']):
                red, green, blue = palette[label]
                file.write(
                    f'{label}\t{class_name}\t{red} {green} {blue}\n')

        # Keep the original RGB point cloud in a separate file.
        from .detection_vis_metric import _write_points_obj
        _write_points_obj(points, scene_dir / f'{scene_name}_points.obj')

    @staticmethod
    def _scene_name(data_sample: dict) -> str:
        eval_ann = data_sample['eval_ann_info']
        point_cloud = eval_ann.get('point_cloud', {})
        if point_cloud.get('lidar_idx', None) is not None:
            return str(point_cloud['lidar_idx'])
        lidar_path = data_sample.get('lidar_path', '')
        if lidar_path:
            return Path(lidar_path).stem
        raise KeyError('The data sample has neither lidar_idx nor lidar_path.')


@METRICS.register_module()
class TR3DSegmentationRankingMetric(SegMetric):
    """Record per-scene segmentation metrics without exporting point clouds."""

    def __init__(self, output_csv: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_csv = Path(output_csv)
        self.scene_rows = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch, data_samples)
        class_names = tuple(self.dataset_meta['classes'])
        num_classes = len(class_names)
        for data_sample in data_samples:
            gt = _numpy(
                data_sample['eval_ann_info']['pts_semantic_mask']).astype(
                    np.int64, copy=False)
            pred = _numpy(
                data_sample['pred_pts_seg']['pts_semantic_mask']).astype(
                    np.int64, copy=False)
            valid = (gt >= 0) & (gt < num_classes)
            valid_gt, valid_pred = gt[valid], pred[valid]
            row = {
                'scene': TR3DSegmentationVisMetric._scene_name(data_sample),
                'total_points': len(gt),
                'valid_points': int(valid.sum()),
                'ignore_fraction': float(1.0 - valid.mean()),
                'accuracy': float(np.mean(valid_gt == valid_pred)),
            }
            class_ious = []
            for label, class_name in enumerate(class_names):
                gt_mask = valid_gt == label
                pred_mask = valid_pred == label
                union = np.count_nonzero(gt_mask | pred_mask)
                iou = (float(np.count_nonzero(gt_mask & pred_mask) / union)
                       if union else float('nan'))
                row[f'iou_{class_name}'] = iou
                row[f'points_{class_name}'] = int(gt_mask.sum())
                if union:
                    class_ious.append(iou)
            row['miou'] = float(np.mean(class_ious))
            row['evaluated_classes'] = len(class_ious)
            self.scene_rows.append(row)

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.scene_rows[0]) if self.scene_rows else []
        with self.output_csv.open('w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.scene_rows)
        return metrics
