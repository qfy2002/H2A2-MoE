# H2A2-MoE: Homogeneity-Aware and Heterogeneity-Aware Feature Perception with Soft Mixture-of-Experts for Unified Indoor 3D Perception

Official evaluation code for **H2A2-MoE**, a unified indoor 3D perception framework for jointly learning 3D object detection and semantic segmentation across heterogeneous indoor datasets.

H2A2-MoE is built upon a TR3D-style sparse 3D backbone and introduces **Variance-Aware Soft Mixture-of-Experts (VA-SoftMoE)** to improve representation sharing and task-specific adaptation. The model supports six indoor 3D detection benchmarks and two semantic segmentation benchmarks within a unified framework.

## Highlights

- **Unified indoor 3D perception:** jointly supports object detection and semantic segmentation across multiple indoor datasets.
- **VA-SoftMoE:** adaptively combines shared and task-specific expert kernels before sparse convolution.
- **Efficient sparse inference:** expert kernels are assembled in the parameter space and require only one sparse convolution.
- **Eight evaluation tasks:** six detection datasets and two semantic segmentation benchmarks.
- **Standalone evaluation:** evaluation uses the official MMEngine/MMDetection3D stack and does not require the custom distributed training framework.

## Supported Benchmarks

| Task | Dataset | Classes | Config |
| --- | --- | ---: | --- |
| Detection | S3DIS Area 5 | 5 | [`configs/s3dis_det.py`](configs/s3dis_det.py) |
| Detection | ScanNet v2 | 18 | [`configs/scannet_det.py`](configs/scannet_det.py) |
| Detection | ARKitScenes | 17 | [`configs/arkitscenes_det.py`](configs/arkitscenes_det.py) |
| Detection | MultiScan | 17 | [`configs/multiscan_det.py`](configs/multiscan_det.py) |
| Detection | 3RScan | 18 | [`configs/3rscan_det.py`](configs/3rscan_det.py) |
| Detection | ScanNet++ | 84 | [`configs/scannetpp_det.py`](configs/scannetpp_det.py) |
| Segmentation | ScanNet v2 | 20 | [`configs/scannet_seg.py`](configs/scannet_seg.py) |
| Segmentation | S3DIS Area 5 | 13 | [`configs/s3dis_seg.py`](configs/s3dis_seg.py) |

## Release Scope

This repository currently provides:

- model definitions for H2A2-MoE and VA-SoftMoE;
- evaluation configurations for all eight dataset-task settings;
- indoor dataset adapters;
- checkpoint loading and evaluation scripts;
- point-cloud prediction and visualization utilities.

The current release focuses on **evaluation and visualization**. The custom distributed multi-task training implementation, including the training launcher, NGH synchronization logic, fine-tuning hooks, and training configurations, is not included.

Official MMEngine is sufficient for evaluating the released checkpoints.

## Installation

### Environment

The following environment was used for validation:

| Component | Version |
| --- | --- |
| Python | 3.8 |
| PyTorch | 1.13.1 |
| torchvision | 0.14.1 |
| CUDA | 11.6 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| MMDetection | 3.2.0 |
| MMDetection3D | 1.4.0 |
| MinkowskiEngine | 0.5.4 |
| NumPy | 1.24.3 |
| SciPy | 1.10.1 |

Linux with an NVIDIA GPU is recommended for full benchmark evaluation.

### Install Dependencies

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
