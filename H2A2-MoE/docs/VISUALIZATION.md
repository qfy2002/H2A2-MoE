# Point-Cloud Visualization

Export a selected scene on the evaluation server without a display. The scene
index refers to the zero-based order in the prepared validation info file.

## Detection

```bash
python projects/TR3D/visualization/export_visualization.py \
  configs/s3dis_det.py checkpoints/s3dis_det.pth \
  --task det --dataset-name s3dis_det \
  --data-root /path/to/prepared/s3dis_det \
  --scene-index 0 --score-thr 0.1 --output-dir outputs/s3dis_det
```

Each scene contains `*_points.obj` (RGB point cloud), `*_gt.obj`, `*_pred.obj`,
and `*_detection.npz` (boxes, labels, and scores after model postprocessing).
`--score-thr` affects the exported prediction OBJ only, not the evaluation score
threshold. Predictions are not matched to ground truth or filtered by IoU.

## Segmentation

```bash
python projects/TR3D/visualization/export_visualization.py \
  configs/scannet_seg.py checkpoints/scannet_seg.pth \
  --task seg --dataset-name scannet_seg \
  --data-root /path/to/prepared/scannet_seg \
  --scene-index 0 --output-dir outputs/scannet_seg
```

The exporter preserves the original points before voxel quantization, then maps
predicted voxel labels back to points. Outputs include RGB points, semantic
predictions, ground truth, and the color palette. Use `--max-scenes 0` to export
the entire split, or `--max-scenes 10` to export the first ten scenes. Dataset
metrics from a scene subset describe that subset only.

## Desktop Viewer

Install `numpy` and `open3d` on the desktop machine. Copy the scene directory and
`projects/TR3D/visualization/visualize_obj.py` there; the viewer does not require
MMDetection3D or a checkpoint.

```bash
python projects/TR3D/visualization/visualize_obj.py /path/to/exported/scene \
  --paper-mode pred --point-size 3.5 --box-style square --box-radius 0.012 \
  --paper-output prediction.png --camera-json camera.json
```

Use the same `--camera-json`, point size, crop options, and score threshold when
comparing methods. `--remove-top`, `--remove-wall`, and `--denoise` affect display
geometry only. Inspect `--help` for options. The GUI viewer requires a display;
only prediction/OBJ export runs without one.
