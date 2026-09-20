# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

from mmengine.model import BaseModule
from torch import Tensor
from torch import nn

from mmdet3d.registry import MODELS
from .minkowski_conv import TR3DMinkowskiConvolutionMulti


def _build_minkowski_conv(in_channels: int,
                          out_channels: int,
                          kernel_size: int,
                          stride: int = 1,
                          dilation: int = 1,
                          dimension: int = 3,
                          conv_type: str = 'standard',
                          conv_cfg: Optional[dict] = None) -> nn.Module:
    """Build a Minkowski convolution used by TR3D local blocks."""
    if ME is None:
        raise ImportError(
            'Please follow `get_started.md` to install MinkowskiEngine.')

    if conv_type == 'standard':
        return ME.MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            dimension=dimension)

    if conv_type == 'multi':
        conv_cfg = conv_cfg or {}
        return TR3DMinkowskiConvolutionMulti(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            **conv_cfg)

    raise ValueError(f'Unsupported TR3D Minkowski conv_type: {conv_type}')


class TR3DMinkBasicBlock(nn.Module):
    """TR3D-local Minkowski basic block.

    This keeps TR3D from depending on patched
    ``MinkowskiEngine.modules.resnet_block.BasicBlock``.
    """

    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 bn_momentum: float = 0.1,
                 dimension: int = 3,
                 conv_type: str = 'standard',
                 conv_cfg: Optional[dict] = None):
        super().__init__()
        assert dimension > 0

        self.conv1 = _build_minkowski_conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            conv_type=conv_type,
            conv_cfg=conv_cfg)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = _build_minkowski_conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
            conv_type=conv_type,
            conv_cfg=conv_cfg)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: SparseTensor) -> SparseTensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class TR3DMinkBottleneck(nn.Module):
    """TR3D-local Minkowski bottleneck block."""

    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 bn_momentum: float = 0.1,
                 dimension: int = 3,
                 conv_type: str = 'standard',
                 conv_cfg: Optional[dict] = None):
        super().__init__()
        assert dimension > 0

        self.conv1 = _build_minkowski_conv(
            inplanes,
            planes,
            kernel_size=1,
            dimension=dimension,
            conv_type='standard')
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv2 = _build_minkowski_conv(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            conv_type=conv_type,
            conv_cfg=conv_cfg)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv3 = _build_minkowski_conv(
            planes,
            planes * self.expansion,
            kernel_size=1,
            dimension=dimension,
            conv_type='standard')
        self.norm3 = ME.MinkowskiBatchNorm(
            planes * self.expansion, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: SparseTensor) -> SparseTensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


@MODELS.register_module()
class TR3DMinkResNet(BaseModule):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input channels, 3 for RGB.
        num_stages (int): Resnet stages. Defaults to 4.
        pool (bool): Whether to add max pooling after first conv.
            Defaults to True.
        norm (str): Norm type ('instance' or 'batch') for stem layer.
            Usually ResNet implies BatchNorm but for some reason
            original MinkResNet implies InstanceNorm. Defaults to 'instance'.
        num_planes (tuple[int]): Number of planes per block before
            block.expansion. Defaults to (64, 128, 256, 512).
        stem_channels (int): Number of output channels in stem conv.
            Defaults to 64.
        stem_kernel_size (int): Kernel size of stem conv. Defaults to 3.
        stem_stride (int): Stride of stem conv. Defaults to 2.
        return_stem (bool): Whether to return stem output before residual
            stages. Detection uses stage outputs only, while segmentation
            needs stem + stages for the decoder. Defaults to False.
        block_conv_type (str): Convolution type inside residual blocks.
            ``'multi'`` uses ``TR3DMinkowskiConvolutionMulti`` locally in
            TR3D. Defaults to 'multi' to match the previous patched ME block.
        block_conv_cfg (dict, optional): Extra args for
            ``TR3DMinkowskiConvolutionMulti``. For example
            ``dict(num_shared_experts=3, num_private_experts=1)``.
    """

    arch_settings = {
        18: (TR3DMinkBasicBlock, (2, 2, 2, 2)),
        34: (TR3DMinkBasicBlock, (3, 4, 6, 3)),
        50: (TR3DMinkBottleneck, (3, 4, 6, 3)),
        101: (TR3DMinkBottleneck, (3, 4, 23, 3)),
        152: (TR3DMinkBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 in_channels: int,
                 num_stages: int = 4,
                 pool: bool = True,
                 norm: str = 'instance',
                 num_planes: Tuple[int] = (64, 128, 256, 512),
                 stem_channels: int = 64,
                 stem_kernel_size: int = 3,
                 stem_stride: int = 2,
                 return_stem: bool = False,
                 block_conv_type: str = 'multi',
                 block_conv_cfg: Optional[dict] = None,
                 stem_conv_type: str = 'standard',
                 downsample_conv_type: str = 'standard'):
        super().__init__()
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.')
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.num_stages = num_stages
        self.pool = pool
        self.return_stem = return_stem
        self.block_conv_type = block_conv_type
        self.block_conv_cfg = block_conv_cfg or {}
        self.downsample_conv_type = downsample_conv_type
        self.inplanes = stem_channels
        self.conv1 = _build_minkowski_conv(
            in_channels,
            stem_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            dimension=3,
            conv_type=stem_conv_type)
        norm_layer = ME.MinkowskiInstanceNorm if norm == 'instance' else \
            ME.MinkowskiBatchNorm
        self.norm1 = norm_layer(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(
                kernel_size=2, stride=2, dimension=3)

        for i in range(len(stage_blocks)):
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(
                    block, num_planes[i], stage_blocks[i], stride=2))

    def init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            elif (hasattr(m, 'shared_experts') or hasattr(m, 'experts')
                  or hasattr(m, 'kernel_sp')):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block: Union[TR3DMinkBasicBlock,
                                      TR3DMinkBottleneck], planes: int,
                    blocks: int, stride: int) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _build_minkowski_conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3,
                    conv_type=self.downsample_conv_type),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3,
                conv_type=self.block_conv_type,
                conv_cfg=self.block_conv_cfg)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dimension=3,
                    conv_type=self.block_conv_type,
                    conv_cfg=self.block_conv_cfg))
        return nn.Sequential(*layers)

    def _should_return_stem(self, task: Optional[str],
                            return_stem: Optional[bool]) -> bool:
        if return_stem is not None:
            return return_stem
        if task in ('seg', 'segmentation', 'semantic_seg'):
            return True
        if task in ('det', 'detection'):
            return False
        return self.return_stem

    def forward(self,
                x: Union[SparseTensor, Tensor],
                coors: Optional[Tensor] = None,
                task: Optional[str] = None,
                return_stem: Optional[bool] = None) -> List[SparseTensor]:
        """Forward pass shared by detection and segmentation.

        Detection keeps the original TR3D behavior and returns residual stage
        outputs. Segmentation asks for the stem output as an additional skip
        connection for the MinkUNet decoder.
        """
        if coors is not None:
            x = ME.SparseTensor(features=x, coordinates=coors)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        outs = [x] if self._should_return_stem(task, return_stem) else []
        if self.pool:
            x = self.maxpool(x)
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs
