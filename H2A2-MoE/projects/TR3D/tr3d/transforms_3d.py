# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints
try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

def _as_numpy(data) -> np.ndarray:
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    return np.asarray(data)



@TRANSFORMS.register_module()
class TR3DKeepPointsForVisualization(BaseTransform):
    """Keep the post-sampling points before sparse quantization."""

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        input_dict['visualization_points'] = points.tensor.numpy().copy()
        return input_dict



@TRANSFORMS.register_module()
class TR3DOfficialSparseQuantize(BaseTransform):
    r"""CPU sparse quantization following MinkowskiEngine data processing.

    It mirrors ``MinkowskiEngine.utils.sparse_quantize``: compute integer voxel
    coordinates with ``floor(coords / voxel_size)``, select one representative
    point per unique coordinate, and keep the corresponding label.
    """

    def __init__(self,
                 voxel_size: float = 0.02,
                 ignore_index: Optional[int] = None,
                 average_features: bool = False) -> None:
        self.voxel_size = voxel_size
        self.ignore_index = ignore_index
        self.average_features = average_features

    def transform(self, input_dict: dict) -> dict:
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.')
        points = input_dict['points']
        assert isinstance(points, BasePoints)
        coords = points.coord.numpy().astype(np.float32, copy=False)

        if 'pts_semantic_mask' in input_dict and self.ignore_index is not None:
            quantized = ME.utils.sparse_quantize(
                coordinates=coords,
                labels=input_dict['pts_semantic_mask'].astype(
                    np.int32, copy=False),
                ignore_label=self.ignore_index,
                return_index=True,
                return_inverse=True,
                quantization_size=self.voxel_size)
            quantized_coords, voxel_labels, inds, inverse = quantized
            input_dict['pts_semantic_mask'] = _as_numpy(voxel_labels).astype(
                np.int64, copy=False)
        else:
            quantized = ME.utils.sparse_quantize(
                coordinates=coords,
                return_index=True,
                return_inverse=True,
                quantization_size=self.voxel_size)
            quantized_coords, inds, inverse = quantized
            if 'pts_semantic_mask' in input_dict:
                input_dict['pts_semantic_mask'] = input_dict[
                    'pts_semantic_mask'][_as_numpy(inds)].astype(
                        np.int64, copy=False)

        inds = _as_numpy(inds).astype(np.int64, copy=False)
        inverse = _as_numpy(inverse).astype(np.int64, copy=False)
        quantized_coords = _as_numpy(quantized_coords).astype(
            np.float32, copy=False)
        point_tensor = points.tensor.numpy()[inds].copy()
        point_tensor[:, :3] = quantized_coords
        if self.average_features and point_tensor.shape[1] > 3:
            num_voxels = quantized_coords.shape[0]
            features = points.tensor.numpy()[:, 3:].astype(
                np.float32, copy=False)
            feature_sum = np.zeros(
                (num_voxels, features.shape[1]), dtype=np.float32)
            np.add.at(feature_sum, inverse, features)
            counts = np.bincount(inverse, minlength=num_voxels).clip(
                min=1).astype(np.float32)
            point_tensor[:, 3:] = feature_sum / counts[:, None]
        input_dict['points'] = type(points)(
            point_tensor,
            points_dim=points.points_dim,
            attribute_dims=points.attribute_dims)

        if 'pts_instance_mask' in input_dict:
            input_dict['pts_instance_mask'] = input_dict['pts_instance_mask'][
                inds]
        input_dict['point2voxel_map'] = inverse
        return input_dict



@TRANSFORMS.register_module()
class TR3DPack3DDetInputs(Pack3DDetInputs):
    """Pack 3D inputs and keep the original point-to-voxel inverse map."""

    def __init__(self, keys: tuple, meta_keys: Optional[tuple] = None) -> None:
        if meta_keys is None:
            meta_keys = Pack3DDetInputs.__init__.__defaults__[0]
        meta_keys = tuple(meta_keys)
        if 'point2voxel_map' not in meta_keys:
            meta_keys = meta_keys + ('point2voxel_map', )
        super().__init__(keys=keys, meta_keys=meta_keys)

