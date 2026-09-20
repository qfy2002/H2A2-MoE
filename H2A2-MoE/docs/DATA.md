# Evaluation Data

These configs consume prepared MMDetection3D info files and binary point clouds,
not the raw dataset downloads. Use the validation split for ScanNet,
ARKitScenes, MultiScan, 3RScan, and ScanNet++; use Area 5 for S3DIS.

The detection adapters follow [UniDet3D data preparation](https://github.com/filapro/unidet3d/tree/main/data).
Its [prepared data collection](https://huggingface.co/datasets/maksimko123/UniDet3D)
is an additional reference. Segmentation uses the
[MMDetection3D ScanNet](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/scannet.html)
and [S3DIS](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/s3dis.html)
preparation conventions. Obtain data under the corresponding dataset terms.

## Roots and Annotation Files

| Config | Default Root | Info File |
| --- | --- | --- |
| `s3dis_det.py` | `data/s3dis_det/` | `s3dis_sp_infos_Area_5.pkl` |
| `scannet_det.py` | `data/scannet_det/` | `scannet_infos_val.pkl` |
| `arkitscenes_det.py` | `data/arkitscenes/` | `arkitscenes_offline_infos_val.pkl` |
| `multiscan_det.py` | `data/multiscan/` | `multiscan_infos_val.pkl` |
| `3rscan_det.py` | `data/3rscan/` | `3rscan_infos_val.pkl` |
| `scannetpp_det.py` | `data/scannetpp/` | `scannetpp_infos_val.pkl` |
| `scannet_seg.py` | `data/scannet_seg/` | `scannet_infos_val_new.pkl` |
| `s3dis_seg.py` | `data/s3dis_seg/` | `s3dis_infos_Area_5.pkl` |

You may point each config at an existing prepared directory. No copying is
necessary:

```bash
python tools/test.py configs/scannet_det.py checkpoints/scannet_det.pth \
  --data-root /path/to/prepared/scannet
```

`--data-root` changes the dataset object's root, including nested datasets.
Changing a top-level `data_root` with `--cfg-options` alone does not rewrite the
already constructed dataset config; use the dedicated option instead.

## File Formats

Info pickles must use MMDetection3D 1.x's `metainfo`/`data_list` format. Detection
entries include `lidar_points.lidar_path` and `instances` containing `bbox_3d`
and `bbox_label_3d`. Keep dataset-specific class ordering and box conventions
unchanged. ScanNet detection also requires the alignment matrix consumed by
`GlobalAlignment`. Do not align the same point cloud a second time.

Binary point files contain float32 `x, y, z, r, g, b`. ARKitScenes input RGB is
expected in [0, 1] and is denormalized by its pipeline; the other included
pipelines expect RGB in [0, 255]. The evaluation configs preserve their original
normalization and sampling behavior.

ARKitScenes expects `offline_prepared_data/` and `super_points/`. MultiScan expects
`points/` and `super_points/`; 3RScan and ScanNet++ expect `points/` and
`super_points_spt/`. Superpoint files contain int64 indices with one entry per
input point. These test pipelines retain superpoint loading for compatibility
with the original preprocessing, even though the TR3D detector itself consumes
points. Set the root at the directory containing the info pickle and these
subdirectories (for some downloads this is the `bins/` directory).

Segmentation roots contain `points/`, `semantic_mask/`, and, where referenced,
`instance_mask/`. Info entries contain semantic-mask paths. Preserve original
semantic IDs for `PointSegClassMapping`; do not map labels twice. ScanNet uses
20 target classes, S3DIS 13. S3DIS's config also retains the
`seg_info/Area_5_resampled_scene_idxs.npy` setting from the original dataset
config; test-mode sampling is handled by the upstream dataset implementation.

The names `scannet_infos_val_new.pkl` and `s3dis_sp_infos_Area_5.pkl` come from the
original experiments. A public preparation tool may use a different filename.
Only replace the filename after checking that its schema, split, point paths,
class order, and coordinate convention match. For example:

```bash
python tools/test.py configs/scannet_seg.py checkpoints/scannet_seg.pth \
  --data-root /path/to/prepared/scannet \
  --cfg-options test_dataloader.dataset.ann_file=scannet_infos_val.pkl
```

This source release does not redistribute datasets or private info files.
