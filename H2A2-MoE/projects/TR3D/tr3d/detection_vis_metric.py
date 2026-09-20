"""Headless OBJ export for TR3D indoor detection predictions."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from mmdet3d.evaluation.metrics import IndoorMetric
from mmdet3d.registry import METRICS


_BOX_FACES = (
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
    (2, 3, 7, 6), (3, 0, 4, 7), (1, 2, 6, 5),
)


def _to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _point_colors(points: np.ndarray) -> Optional[np.ndarray]:
    if points.shape[1] < 6:
        return None
    colors = points[:, 3:6].astype(np.float32, copy=True)
    # TR3D configurations use either [0, 1] normalization or [-1, 1].
    if colors.min(initial=0.0) < -0.05:
        colors = (colors + 1.0) * 127.5
    else:
        colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def _write_points_obj(points: np.ndarray, path: Path) -> None:
    colors = _point_colors(points)
    with path.open('w') as file:
        if colors is None:
            for point in points:
                file.write(f'v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n')
        else:
            for point, color in zip(points, colors):
                file.write(
                    f'v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} '
                    f'{int(color[0])} {int(color[1])} {int(color[2])}\n')


def _label_color(label: int) -> tuple:
    # Stable colors make overlapping classes distinguishable in OBJ viewers.
    palette = (
        (230, 75, 53), (36, 132, 198), (60, 179, 113),
        (243, 156, 18), (155, 89, 182), (22, 160, 133),
    )
    return palette[int(label) % len(palette)]


def _write_boxes_obj(corners: np.ndarray, labels: np.ndarray, path: Path) -> None:
    with path.open('w') as file:
        for index, (box, label) in enumerate(zip(corners, labels)):
            color = _label_color(int(label))
            for point in box:
                file.write(
                    f'v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} '
                    f'{color[0]} {color[1]} {color[2]}\n')
            offset = index * 8 + 1
            for face in _BOX_FACES:
                file.write('f ' + ' '.join(str(offset + vertex) for vertex in face) + '\n')


@METRICS.register_module()
class TR3DDetectionVisMetric(IndoorMetric):
    """Evaluate indoor detection and export point clouds and boxes as OBJ.

    The exporter reads points from the evaluator's input batch instead of the
    prediction. This keeps the base ``MinkSingleStage3DDetector`` unchanged.
    """

    def __init__(self,
                 vis_dir: str,
                 dataset_name: str = 'unknown',
                 score_thr: float = 0.1,
                 max_samples: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.vis_dir = Path(vis_dir)
        self.dataset_name = dataset_name
        self.score_thr = score_thr
        self.max_samples = max_samples
        self._exported_samples = 0

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch, data_samples)
        points_list = data_batch.get('inputs', {}).get('points', [])
        for points, data_sample in zip(points_list, data_samples):
            if self.max_samples is not None and self._exported_samples >= self.max_samples:
                break
            self._export_scene(_to_numpy(points), data_sample)
            self._exported_samples += 1

    def _export_scene(self, points: np.ndarray, data_sample: dict) -> None:
        scene_name = self._scene_name(data_sample)
        scene_dir = self.vis_dir / self.dataset_name / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        eval_ann = data_sample['eval_ann_info']
        gt_boxes = _to_numpy(eval_ann['gt_bboxes_3d'].corners)
        gt_labels = _to_numpy(eval_ann['gt_labels_3d'])

        pred = data_sample['pred_instances_3d']
        pred_boxes = _to_numpy(pred['bboxes_3d'].corners)
        pred_labels = _to_numpy(pred['labels_3d'])
        scores = _to_numpy(pred['scores_3d'])
        keep = scores >= self.score_thr

        # Save unfiltered detections so different models can be compared with
        # exactly the same score threshold without rerunning inference.
        np.savez_compressed(
            scene_dir / f'{scene_name}_detection.npz',
            gt_corners=gt_boxes,
            gt_labels=gt_labels,
            pred_corners=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=scores,
            point_min=points[:, :3].min(axis=0),
            point_max=points[:, :3].max(axis=0),
            num_points=np.asarray(points.shape[0], dtype=np.int64))

        _write_points_obj(points, scene_dir / f'{scene_name}_points.obj')
        _write_boxes_obj(gt_boxes, gt_labels, scene_dir / f'{scene_name}_gt.obj')
        _write_boxes_obj(pred_boxes[keep], pred_labels[keep],
                         scene_dir / f'{scene_name}_pred.obj')

    @staticmethod
    def _scene_name(data_sample: dict) -> str:
        lidar_idx = data_sample.get('lidar_idx', None)
        if lidar_idx is not None:
            return str(lidar_idx)
        lidar_path = data_sample.get('lidar_path', '')
        if lidar_path:
            return Path(lidar_path).stem
        raise KeyError('The data sample has neither lidar_idx nor lidar_path.')
