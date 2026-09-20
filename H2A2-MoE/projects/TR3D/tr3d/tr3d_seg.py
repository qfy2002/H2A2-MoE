# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from mmengine.model import BaseModule
from torch import Tensor, nn

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    ME = SparseTensor = None

from mmdet3d.models.decode_heads import Base3DDecodeHead
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.segmentors import MinkUNet
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .mink_resnet import (TR3DMinkBasicBlock, TR3DMinkBottleneck,
                          TR3DMinkResNet)


@MODELS.register_module()
class TR3DOfficialMinkowskiPreprocessor(Det3DDataPreprocessor):
    r"""MinkowskiEngine-style sparse quantization for TR3D segmentation.

    NVIDIA/MinkowskiEngine's segmentation utility uses
    ``MinkowskiEngine.utils.sparse_quantize`` to build a unique map and an
    inverse map for quantized coordinates. This preprocessor mirrors that
    unique-map behavior while keeping the point-to-voxel map required by
    MMDetection3D segmentation evaluation.
    """

    def __init__(self, ignore_index: int = -100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def _get_voxel_semantic_mask(self, pts_semantic_mask: Tensor, inds: Tensor,
                                 point2voxel_map: Tensor,
                                 num_voxels: int) -> Tensor:
        """Match ME quantize_label: conflicting labels become ignore."""
        if self.ignore_index < 0:
            return pts_semantic_mask[inds]

        num_labels = self.ignore_index + 1
        labels = pts_semantic_mask.long().clamp(min=0, max=self.ignore_index)
        label_hist = labels.new_zeros((num_voxels, num_labels))
        label_hist.index_add_(
            0, point2voxel_map.long(),
            F.one_hot(labels, num_classes=num_labels).to(label_hist.dtype))
        voxel_semantic_mask = label_hist.argmax(dim=1).to(
            pts_semantic_mask.dtype)
        voxel_semantic_mask[label_hist.gt(0).sum(dim=1).ne(1)] = \
            self.ignore_index
        return voxel_semantic_mask

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> dict:
        if self.voxel_type == 'prequantized_minkunet':
            voxels, coors = [], []
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                res_voxel_coors = res[:, :3].int()
                res_voxels = res
                if hasattr(data_sample, 'gt_pts_seg') and hasattr(
                        data_sample.gt_pts_seg, 'pts_semantic_mask'):
                    data_sample.gt_pts_seg.voxel_semantic_mask = \
                        data_sample.gt_pts_seg.pts_semantic_mask

                point2voxel_map = data_sample.metainfo.get(
                    'point2voxel_map', None)
                if point2voxel_map is None:
                    point2voxel_map = torch.arange(
                        res_voxels.shape[0], device=res_voxels.device)
                else:
                    point2voxel_map = torch.as_tensor(
                        point2voxel_map, device=res_voxels.device)
                if self.batch_first:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (1, 0), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, 0]
                else:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (0, 1), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, -1]
                if 'point2voxel_map' in data_sample.metainfo:
                    data_sample.set_metainfo(
                        {'point2voxel_map': point2voxel_map.long()})
                else:
                    data_sample.point2voxel_map = point2voxel_map.long()
                voxels.append(res_voxels)
                coors.append(res_voxel_coors)
            return dict(
                voxels=torch.cat(voxels, dim=0), coors=torch.cat(coors))

        if self.voxel_type != 'minkunet':
            return super().voxelize(points, data_samples)

        voxels, coors = [], []
        voxel_size = points[0].new_tensor(self.voxel_layer.voxel_size)
        for i, (res, data_sample) in enumerate(zip(points, data_samples)):
            res_coors = torch.floor(res[:, :3] / voxel_size).int()

            inds, point2voxel_map = self.sparse_quantize(
                res_coors.cpu().numpy(),
                return_index=True,
                return_inverse=True)
            inds = torch.from_numpy(inds).to(res.device)
            point2voxel_map = torch.from_numpy(point2voxel_map).to(res.device)
            res_voxel_coors = res_coors[inds]
            res_voxels = res[inds].clone()
            if not self.training and res_voxels.shape[1] > 3:
                # ME training examples use sparse_quantize representative
                # features; indoor.py uses TensorField UNWEIGHTED_AVERAGE.
                num_voxels = inds.numel()
                counts = torch.bincount(
                    point2voxel_map, minlength=num_voxels).clamp_min(1)
                feature_sum = res.new_zeros(num_voxels,
                                            res_voxels.shape[1] - 3)
                feature_sum.index_add_(0, point2voxel_map, res[:, 3:])
                res_voxels[:, 3:] = feature_sum / counts.to(
                    res.dtype)[:, None]
            if hasattr(data_sample, 'gt_pts_seg') and hasattr(
                    data_sample.gt_pts_seg, 'pts_semantic_mask'):
                data_sample.gt_pts_seg.voxel_semantic_mask = \
                    self._get_voxel_semantic_mask(
                        data_sample.gt_pts_seg.pts_semantic_mask, inds,
                        point2voxel_map.long(), inds.numel())

            if self.batch_first:
                res_voxel_coors = F.pad(
                    res_voxel_coors, (1, 0), mode='constant', value=i)
                data_sample.batch_idx = res_voxel_coors[:, 0]
            else:
                res_voxel_coors = F.pad(
                    res_voxel_coors, (0, 1), mode='constant', value=i)
                data_sample.batch_idx = res_voxel_coors[:, -1]
            data_sample.point2voxel_map = point2voxel_map.long()
            voxels.append(res_voxels)
            coors.append(res_voxel_coors)

        return dict(voxels=torch.cat(voxels, dim=0), coors=torch.cat(coors))


@MODELS.register_module()
class TR3DMinkSegResNet(TR3DMinkResNet):
    r"""TR3D Minkowski ResNet encoder for semantic segmentation.

    The detection TR3D backbone returns four high-level feature maps. The
    official MinkowskiEngine MinkUNet decoder needs five tensors
    ``(p1, p2, p4, p8, p16)`` for skip connections, so this segmentation
    variant exposes the stem output together with the four residual stages.
    """

    def __init__(self,
                 depth: int = 34,
                 in_channels: int = 3,
                 num_stages: int = 4,
                 pool: bool = False,
                 norm: str = 'batch',
                 num_planes: Tuple[int, ...] = (64, 128, 128, 128),
                 stem_channels: int = 64,
                 stem_stride: int = 1):
        super().__init__(
            depth=depth,
            in_channels=in_channels,
            num_stages=num_stages,
            pool=pool,
            norm=norm,
            num_planes=num_planes,
            stem_channels=stem_channels,
            stem_kernel_size=5,
            stem_stride=stem_stride,
            return_stem=True,
            block_conv_type='standard')

    def forward(self, voxel_features: Tensor,
                coors: Tensor) -> List[SparseTensor]:
        """Forward the sparse voxels and return stem + stage features."""
        return super().forward(voxel_features, coors, task='seg')


@MODELS.register_module()
class TR3DMinkUNetDecoder(BaseModule):
    r"""MinkowskiEngine official MinkUNet decoder used after TR3D encoder.

    This mirrors the decoder part of NVIDIA/MinkowskiEngine
    ``examples/minkunet.py``: four transpose convolutions with BN/ReLU, skip
    concatenation, and residual blocks 5-8.
    """

    arch_settings = {
        14: (TR3DMinkBasicBlock, (1, 1, 1, 1)),
        18: (TR3DMinkBasicBlock, (2, 2, 2, 2)),
        34: (TR3DMinkBasicBlock, (2, 2, 2, 2)),
        50: (TR3DMinkBottleneck, (2, 2, 2, 2)),
        101: (TR3DMinkBottleneck, (2, 2, 2, 2))
    }

    def __init__(self,
                 depth: int = 34,
                 encoder_channels: Sequence[int] = (32, 32, 64, 128, 256),
                 planes: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
                 out_channels: Optional[int] = None,
                 bn_momentum: float = 0.1,
                 dimension: int = 3):
        super().__init__()
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.')
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for MinkUNet decoder')
        assert len(encoder_channels) == 5
        assert len(planes) == 8
        self.D = dimension
        self.BLOCK, decoder_layers = self.arch_settings[depth]
        self.PLANES = tuple(planes)
        self.encoder_channels = tuple(encoder_channels)
        self.bn_momentum = bn_momentum
        self.out_channels = out_channels or self.PLANES[7] * self.BLOCK.expansion

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            encoder_channels[4],
            self.PLANES[4],
            kernel_size=2,
            stride=2,
            dimension=dimension)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.inplanes = self.PLANES[4] + encoder_channels[3]
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       decoder_layers[0])

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.PLANES[5],
            kernel_size=2,
            stride=2,
            dimension=dimension)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.inplanes = self.PLANES[5] + encoder_channels[2]
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       decoder_layers[1])

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.PLANES[6],
            kernel_size=2,
            stride=2,
            dimension=dimension)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.inplanes = self.PLANES[6] + encoder_channels[1]
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       decoder_layers[2])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes,
            self.PLANES[7],
            kernel_size=2,
            stride=2,
            dimension=dimension)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.inplanes = self.PLANES[7] + encoder_channels[0]
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       decoder_layers[3])
        self.relu = ME.MinkowskiReLU(inplace=True)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (ME.MinkowskiConvolution,
                              ME.MinkowskiConvolutionTranspose)):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self,
                    block: nn.Module,
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilation: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                bn_momentum=self.bn_momentum,
                dimension=self.D)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    bn_momentum=self.bn_momentum,
                    dimension=self.D))
        return nn.Sequential(*layers)

    def forward(self, x: List[SparseTensor]) -> SparseTensor:
        assert len(x) == 5, 'TR3DMinkUNetDecoder expects p1,p2,p4,p8,p16'
        out_p1, out_b1p2, out_b2p4, out_b3p8, out = x

        out = self.relu(self.bntr4(self.convtr4p16s2(out)))
        out = self.block5(ME.cat(out, out_b3p8))

        out = self.relu(self.bntr5(self.convtr5p8s2(out)))
        out = self.block6(ME.cat(out, out_b2p4))

        out = self.relu(self.bntr6(self.convtr6p4s2(out)))
        out = self.block7(ME.cat(out, out_b1p2))

        out = self.relu(self.bntr7(self.convtr7p2s2(out)))
        out = self.block8(ME.cat(out, out_p1))
        return out


@MODELS.register_module()
class TR3DMinkUNetHead(Base3DDecodeHead):
    r"""Official MinkowskiEngine MinkUNet 1x1 sparse convolution head."""

    def __init__(self, channels: int, num_classes: int, **kwargs) -> None:
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.')
        super().__init__(channels, num_classes, dropout_ratio=0, **kwargs)

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return ME.MinkowskiConvolution(
            channels,
            num_classes,
            kernel_size=kernel_size,
            bias=True,
            dimension=3)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        return torch.cat(gt_semantic_segs)

    def init_weights(self) -> None:
        BaseModule.init_weights(self)
        if isinstance(self.conv_seg, ME.MinkowskiConvolution):
            ME.utils.kaiming_normal_(
                self.conv_seg.kernel, mode='fan_out', nonlinearity='relu')
            if self.conv_seg.bias is not None:
                nn.init.constant_(self.conv_seg.bias, 0)

    def forward(self, x: SparseTensor) -> Tensor:
        return self.conv_seg(x).F

    def predict(self, inputs: SparseTensor,
                batch_data_samples: SampleList) -> List[Tensor]:
        seg_logits = self.forward(inputs)
        batch_idx = torch.cat(
            [data_sample.batch_idx for data_sample in batch_data_samples])
        seg_logit_list = []
        for i, data_sample in enumerate(batch_data_samples):
            seg_logit = seg_logits[batch_idx == i]
            seg_logit = seg_logit[data_sample.point2voxel_map]
            seg_logit_list.append(seg_logit)
        return seg_logit_list


@MODELS.register_module()
class TR3DMinkUNet(MinkUNet):
    r"""MinkUNet-style segmentor with a TR3D Minkowski ResNet encoder."""

    def __init__(self,
                 input_feature_indices: Optional[Sequence[int]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_feature_indices = input_feature_indices

    def extract_feat(self, batch_inputs_dict: dict) -> SparseTensor:
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = voxel_dict['voxels']
        if self.input_feature_indices is not None:
            voxel_features = voxel_features[:,
                                            list(self.input_feature_indices)]
        x = self.backbone(voxel_features, voxel_dict['coors'], task='seg')
        if self.with_neck:
            x = self.neck(x)
        return x
