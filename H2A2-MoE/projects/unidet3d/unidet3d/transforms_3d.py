# Adapted from UniDet3D; see projects/unidet3d/LICENSE.
import numpy as np
from mmdet3d.datasets.transforms import PointSample
from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PointSample_(PointSample):

    def _points_random_sampling(self, points, num_samples):
        """Points random sampling. Sample points to a certain number.
        
        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:
                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """

        point_range = range(len(points))
        choices = np.random.choice(point_range, 
                                   min(num_samples, len(points)))
        
        return points[choices], choices

    def transform(self, input_dict):
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask',
            'pts_semantic_mask', sp_pts_mask' keys are updated in the 
            result dict.
        """
        points = input_dict['points']
        sp_mask = input_dict.get('sp_pts_mask')
        if sp_mask is not None and sp_mask.shape[0] != points.shape[0]:
            raise ValueError('Superpoint mask and point cloud have different lengths.')
        points, choices = self._points_random_sampling(
            points, self.num_points)
        input_dict['points'] = points
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            
            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask
        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask
        return input_dict
