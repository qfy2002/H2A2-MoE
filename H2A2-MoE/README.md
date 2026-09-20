# H2A2-MoE: Unified Indoor 3D Perception

Evaluation code for H2A2-MoE, covering six indoor 3D detection benchmarks and
two semantic segmentation benchmarks. The model uses a TR3D-style sparse
backbone with variance-aware soft mixtures of expert kernels (VA-SoftMoE).

This release includes model definitions, evaluation configurations, dataset
adapters, and point-cloud visualization. It does **not** include the custom
MMEngine training implementation, multi-task training launcher, NGH optimizer,
fine-tuning hooks, or training configurations. Official MMEngine is sufficient
for evaluation. Loss modules required by the upstream model constructors remain
in the model package; they are not invoked during testing.

## Installation

Use Linux with an NVIDIA GPU for full evaluation. Detection uses MMCV's CUDA NMS
operators; CPU model construction is supported, but is not a replacement for
benchmark evaluation. A desktop display is needed only for the optional OBJ
viewer.

Run commands from this repository's root. The evaluation scripts add this
repository to the import path automatically; no overlay onto an existing
MMDetection3D checkout is required.

### Reference Versions

| Component | Version |
| --- | --- |
| Python | 3.8 |
| PyTorch / torchvision | 1.13.1 / 0.14.1 |
| CUDA build used for validation | 11.6 |
| MMCV | 2.1.0 (with compiled operators) |
| MMEngine | 0.10.7 (official package) |
| MMDetection | 3.2.0 |
| MMDetection3D | 1.4.0 |
| MinkowskiEngine | 0.5.4 |
| NumPy / SciPy | 1.24.3 / 1.10.1 |

### Install Dependencies

Create a separate environment. This example uses PyTorch's CUDA 11.6 build;
the locally installed CUDA toolkit used for compilation must match it.

```bash
conda create -n h2a2_moe_eval python=3.8 -y
conda activate h2a2_moe_eval
python -m pip install 'pip<25' 'setuptools<70' wheel ninja
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install numpy==1.24.3 scipy==1.10.1
python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
python -m pip install -r requirements.txt
```

Do not substitute `mmcv-lite`: 3D NMS requires compiled MMCV operators. If a wheel
is not available for your platform, follow the upstream MMCV build instructions
with the same PyTorch and CUDA versions.

Install official MMEngine rather than copying the custom MMEngine installation
used for joint training. Training process groups and gradient synchronization
are not needed to evaluate a task checkpoint.

### MinkowskiEngine

Install the standard MinkowskiEngine 0.5.4 implementation; no custom convolution
or ResNet patch inside the library is needed. The model's custom convolution and
residual blocks are implemented in this repository.

```bash
conda install -c conda-forge openblas-devel -y
git clone --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas=openblas --force_cuda
cd ..
```

Point `CUDA_HOME` at your CUDA toolkit if it is not found automatically. Use a
compiler supported by that toolkit. When building on a host without a visible
GPU, also set `TORCH_CUDA_ARCH_LIST` for the GPU architecture where evaluation
will run. See the upstream installation instructions for compiler-specific
issues. These dependencies are external; their source trees are not part of
this release.

### Verify Installation

```bash
OMP_NUM_THREADS=2 python tools/check_install.py
python tools/test.py --help
```

The check builds all eight configurations and their test transforms on CPU,
without accessing dataset files. To check all downloaded checkpoint shapes and
keys as well:

```bash
python tools/check_install.py --checkpoints checkpoints
```

## Data and Weights

Prepare the datasets following [docs/DATA.md](docs/DATA.md). Set the `data_root`
at the top of the desired config, or pass `--data-root` to the evaluation script.
Point clouds and annotations are distributed separately from the source code.
The local release workspace also contains eight exported checkpoints under
`checkpoints/`; they are ignored by Git and excluded from the source ZIP.
Public checkpoint download links are **not yet provided**. Place each weight file
under `checkpoints/` using the filenames listed below.

Each dataset-task slice needs its own checkpoint, which contains its shared
backbone parameters and task-specific components. One task's checkpoint cannot
be substituted for another's, even when their class counts are equal.

| Task | Config | Classes | Expected Checkpoint |
| --- | --- | ---: | --- |
| S3DIS detection (Area 5) | [s3dis_det.py](configs/s3dis_det.py) | 5 | `s3dis_det.pth` |
| ScanNet detection | [scannet_det.py](configs/scannet_det.py) | 18 | `scannet_det.pth` |
| ARKitScenes detection | [arkitscenes_det.py](configs/arkitscenes_det.py) | 17 | `arkitscenes_det.pth` |
| MultiScan detection | [multiscan_det.py](configs/multiscan_det.py) | 17 | `multiscan_det.pth` |
| 3RScan detection | [3rscan_det.py](configs/3rscan_det.py) | 18 | `3rscan_det.pth` |
| ScanNet++ detection | [scannetpp_det.py](configs/scannetpp_det.py) | 84 | `scannetpp_det.pth` |
| ScanNet segmentation | [scannet_seg.py](configs/scannet_seg.py) | 20 | `scannet_seg.pth` |
| S3DIS segmentation (Area 5) | [s3dis_seg.py](configs/s3dis_seg.py) | 13 | `s3dis_seg.pth` |

## Evaluation

Select an available GPU with `CUDA_VISIBLE_DEVICES`. Each command evaluates one
task independently; eight GPUs are not required.

```bash
python tools/test.py configs/s3dis_det.py checkpoints/s3dis_det.pth
python tools/test.py configs/scannet_det.py checkpoints/scannet_det.pth
python tools/test.py configs/arkitscenes_det.py checkpoints/arkitscenes_det.pth
python tools/test.py configs/multiscan_det.py checkpoints/multiscan_det.pth
python tools/test.py configs/3rscan_det.py checkpoints/3rscan_det.pth
python tools/test.py configs/scannetpp_det.py checkpoints/scannetpp_det.pth
python tools/test.py configs/scannet_seg.py checkpoints/scannet_seg.pth
python tools/test.py configs/s3dis_seg.py checkpoints/s3dis_seg.pth
```

The equivalent shell shortcut accepts the task name and checkpoint:

```bash
bash tools/test.sh s3dis_det checkpoints/s3dis_det.pth \
  --data-root /path/to/prepared/s3dis_det --work-dir work_dirs/s3dis_det
```

For a single-scene smoke test, append
`--cfg-options test_dataloader.dataset.indices=1`.
Omit `indices` for benchmark evaluation. `IndoorMetric` reports detection AP;
`SegMetric` reports semantic mIoU and accuracy.

Keep the configured `batch_size=1`. VA-SoftMoE computes routing statistics over
all active voxels in the local batch, so changing the batch composition can change
predictions. Test-time point sampling is retained from the source configurations.
By default, MMEngine generates a fresh random seed for each run and records it
in the log. Pass `--seed INTEGER` to fix the seed for a repeatable comparison.
Different seeds change test-time sampling and can change detection results;
compare repeated-trial statistics when reproducing reported means and maxima.
Score/NMS thresholds and color normalization are preserved.

The test entry point uses strict checkpoint loading. A missing key or mismatched
shape is an error, rather than silently evaluating a partially initialized model.

The eight local checkpoint files are weights-only exports and each passed strict
loading with its matching config. Publish them as separate GitHub Release assets
because they are too large for a normal source-code commit.

## Visualization

Use [docs/VISUALIZATION.md](docs/VISUALIZATION.md) to export RGB points,
predictions, and ground truth without a display, then view them with Open3D.

## Repository Layout

```text
configs/                         Eight standalone test configs and test runtime
projects/TR3D/tr3d/               Sparse backbone, VA-SoftMoE, heads, transforms
projects/TR3D/visualization/      Headless export and desktop OBJ viewer
projects/unidet3d/unidet3d/       Required indoor dataset adapters
tools/                           Evaluation, dependency checks, weight export
docs/                            Dataset preparation and visualization
checkpoints/                     Model weights (distributed separately)
```

## Acknowledgments and Licenses

This implementation builds on [TR3D](https://github.com/SamsungLabs/tr3d),
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d),
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), and the dataset
adapters from [UniDet3D](https://github.com/filapro/unidet3d).

The repository includes components with different license terms. In particular,
the UniDet3D-derived adapters retain their **CC BY-NC 4.0** license; they are not
covered by the root Apache license. See [NOTICE.md](NOTICE.md) for attribution and
the scope of each license.
