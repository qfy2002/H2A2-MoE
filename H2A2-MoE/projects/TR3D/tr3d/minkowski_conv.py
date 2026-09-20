# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

try:
    import MinkowskiEngine as ME
    from MinkowskiConvolution import (MinkowskiConvolutionFunction,
                                      MinkowskiConvolutionTransposeFunction)
    from MinkowskiCoordinateManager import CoordinateManager
    from MinkowskiEngineBackend._C import (ConvolutionMode, CoordinateMapKey,
                                           RegionType)
    from MinkowskiKernelGenerator import KernelGenerator
    from MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
except ImportError:
    ME = None
    MinkowskiConvolutionFunction = MinkowskiConvolutionTransposeFunction = None
    CoordinateManager = CoordinateMapKey = RegionType = ConvolutionMode = None
    KernelGenerator = SparseTensor = _get_coordinate_map_key = None


class TR3DMinkowskiConvolutionMulti(nn.Module):
    """TR3D-local dynamic Minkowski convolution.

    The expert bank is explicitly split into shared and task-specific parts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = -1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        kernel_generator: KernelGenerator = None,
        is_transpose: bool = False,
        expand_coordinates: bool = False,
        convolution_mode: ConvolutionMode = None,
        dimension: int = -1,
        num_experts: int = 4,
        num_shared_experts: Optional[int] = None,
        num_private_experts: Optional[int] = None,
        task_aware: bool = True,
        min_temperature: float = 0.05,
        max_temperature: float = 5.0,
    ):
        super().__init__()
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.')
        assert dimension > 0, (
            'Invalid dimension. Please provide a valid dimension argument. '
            f'dimension={dimension}')

        if convolution_mode is None:
            convolution_mode = ConvolutionMode.DEFAULT
        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                expand_coordinates=expand_coordinates,
                dimension=dimension)
        else:
            kernel_generator.expand_coordinates = expand_coordinates

        self.is_transpose = is_transpose
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_mm = False

        if num_shared_experts is None and num_private_experts is None:
            num_shared_experts = num_experts
            num_private_experts = 0
        elif num_shared_experts is None:
            num_shared_experts = num_experts - num_private_experts
        elif num_private_experts is None:
            num_private_experts = num_experts - num_shared_experts

        self.num_shared_experts = int(num_shared_experts)
        self.num_private_experts = int(num_private_experts)
        if self.num_shared_experts <= 0:
            raise ValueError('num_shared_experts must be positive.')
        if self.num_private_experts < 0:
            raise ValueError('num_private_experts must be non-negative.')
        self.num_experts = self.num_shared_experts + self.num_private_experts
        if self.num_experts <= 0:
            raise ValueError('At least one expert is required.')
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        if (self.kernel_generator.kernel_volume == 1
                and self.kernel_generator.requires_strided_coordinates):
            self.use_mm = True

        self.kernel_volume = self.kernel_generator.kernel_volume
        shared_expert_shape = (self.num_shared_experts, self.kernel_volume,
                               self.in_channels, self.out_channels)
        self.shared_experts = Parameter(torch.empty(*shared_expert_shape))
        if self.num_private_experts > 0:
            private_expert_shape = (self.num_private_experts,
                                    self.kernel_volume, self.in_channels,
                                    self.out_channels)
            self.private_experts = Parameter(torch.empty(*private_expert_shape))
        else:
            self.register_parameter('private_experts', None)

        struct_feat_dim = 64
        self.struct_feat_dim = struct_feat_dim
        self.s_ot = Parameter(torch.empty(self.kernel_volume))
        self.f_ot = Parameter(
            torch.empty(self.kernel_volume, struct_feat_dim))

        self.task_aware = task_aware
        struct_in_channels = in_channels * 2 if self.task_aware else in_channels
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_in_channels, struct_feat_dim),
            nn.LayerNorm(struct_feat_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(struct_feat_dim, struct_feat_dim),
        )
        self.moe_router = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, self.num_experts),
        )
        self.tau = Parameter(torch.tensor(1.0))

        self.bias = Parameter(torch.FloatTensor(1, out_channels)) \
            if bias else None
        self.convolution_mode = convolution_mode
        self.conv = MinkowskiConvolutionTransposeFunction() \
            if is_transpose else MinkowskiConvolutionFunction()

        if self.task_aware:
            self.task_embedding = Parameter(torch.empty(in_channels))
            self.task_mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.Tanh(),
                nn.Linear(in_channels, in_channels * 2),
            )
            self.task_gate = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Sigmoid(),
            )
            self.task_bn = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Tanh(),
            )
            self.task_norm = nn.BatchNorm1d(out_channels)

        self.reset_parameters(is_transpose=is_transpose)

    def _expert_bank(self) -> torch.Tensor:
        if self.private_experts is None:
            return self.shared_experts
        return torch.cat([self.shared_experts, self.private_experts], dim=0)

    def forward(
        self,
        input: SparseTensor,
        coordinates: Union[torch.Tensor, CoordinateMapKey, SparseTensor] = None,
    ) -> SparseTensor:
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        f_in = input.F
        num_points = f_in.shape[0]
        if self.task_aware:
            task_params = self.task_mlp(self.task_embedding)
            scale, shift = torch.split(task_params, self.in_channels)
            f_in = f_in * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)

        if self.task_aware:
            task_emb = self.task_embedding.unsqueeze(0).expand(num_points, -1)
            struct_input = torch.cat([f_in, task_emb], dim=1)
        else:
            struct_input = f_in
        f_struct = self.struct_mlp(struct_input)
        f_struct = F.normalize(f_struct, p=2, dim=1, eps=1e-6)
        f_proto = F.normalize(self.f_ot, p=2, dim=1, eps=1e-6)

        alpha = torch.sigmoid(self.s_ot)
        sim = torch.matmul(f_struct, f_proto.T)
        beta_mean = (sim.mean(dim=0) + 1) / 2
        if num_points > 1:
            beta_std = torch.std(sim, dim=0, unbiased=False)
        else:
            beta_std = torch.zeros_like(beta_mean)

        router_in = torch.stack([alpha, beta_mean, beta_std], dim=-1)
        expert_logits = self.moe_router(router_in)
        tau = torch.clamp(
            self.tau, min=self.min_temperature, max=self.max_temperature)
        if self.training:
            routing_weights = F.gumbel_softmax(
                expert_logits, tau=tau, hard=False, dim=-1)
        else:
            routing_weights = F.softmax(expert_logits / tau, dim=-1)
        kernel = torch.einsum('ve,evio->vio', routing_weights,
                              self._expert_bank())

        if self.use_mm:
            out_coordinate_map_key = input.coordinate_map_key
            kernel_mm = kernel.view(self.in_channels, self.out_channels)
            outfeat = f_in.mm(kernel_mm)
        else:
            out_coordinate_map_key = _get_coordinate_map_key(
                input, coordinates, self.kernel_generator.expand_coordinates)
            outfeat = self.conv.apply(
                f_in,
                kernel,
                self.kernel_generator,
                self.convolution_mode,
                input.coordinate_map_key,
                out_coordinate_map_key,
                input._manager,
            )

        if self.task_aware:
            gate = self.task_gate(self.task_embedding).unsqueeze(0)
            outfeat = outfeat * (1 + gate)
            task_norm_params = self.task_bn(self.task_embedding).unsqueeze(0)
            outfeat = self.task_norm(outfeat)
            outfeat = outfeat * (1 + task_norm_params)

        if self.bias is not None:
            outfeat += self.bias

        return SparseTensor(
            outfeat,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=input._manager,
        )

    def reset_parameters(self, is_transpose: bool = False) -> None:
        with torch.no_grad():
            n = (self.out_channels if is_transpose else self.in_channels) * \
                self.kernel_generator.kernel_volume
            stdv = 1.0 / math.sqrt(n)

            self.shared_experts.data.uniform_(-stdv, stdv)
            if self.private_experts is not None:
                self.private_experts.data.uniform_(-stdv, stdv)
            self.s_ot.data.zero_()
            self.f_ot.data.normal_(0, 0.1)
            self.tau.data.fill_(1.0)

            for m in self.moe_router.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()

            if self.task_aware:
                self.task_embedding.data.normal_(0, 0.1)
                task_mlp_layers = [
                    m for m in self.task_mlp.modules()
                    if isinstance(m, nn.Linear)
                ]
                for m in task_mlp_layers:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            m.bias.data.zero_()
                if task_mlp_layers:
                    task_mlp_layers[-1].weight.data.zero_()
                    if task_mlp_layers[-1].bias is not None:
                        task_mlp_layers[-1].bias.data.zero_()
                for m in self.task_gate.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data.zero_()
                        if m.bias is not None:
                            m.bias.data.fill_(-5.0)
                for m in self.task_bn.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data.zero_()
                        if m.bias is not None:
                            m.bias.data.zero_()
                self.task_norm.reset_parameters()

            for m in self.struct_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()

            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self) -> str:
        s = ('(in={}, out={}, num_shared_experts={}, '
             'num_private_experts={}, ').format(
                 self.in_channels, self.out_channels,
                 self.num_shared_experts, self.num_private_experts)
        if self.kernel_generator.region_type in [RegionType.CUSTOM]:
            s += 'region_type={}, kernel_volume={}, '.format(
                self.kernel_generator.region_type,
                self.kernel_generator.kernel_volume)
        else:
            s += 'kernel_size={}, '.format(self.kernel_generator.kernel_size)
        s += 'stride={}, dilation={})'.format(
            self.kernel_generator.kernel_stride,
            self.kernel_generator.kernel_dilation,
        )
        return self.__class__.__name__ + s
