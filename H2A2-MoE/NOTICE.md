# Attribution and License Scope

- `projects/TR3D/tr3d/` contains TR3D/MMDetection3D-derived model definitions
  and the H2A2-MoE extensions. Existing OpenMMLab copyright headers are retained.
  The Apache 2.0 text is included as `LICENSE` and
  `licenses/MMDetection3D-Apache-2.0.txt`.
- `projects/unidet3d/` contains dataset and preprocessing components adapted from
  [UniDet3D](https://github.com/filapro/unidet3d) by Maksim Kolodiazhnyi,
  Anna Vorontsova, Matvey Skripkin, Danila Rukhovich, and Anton Konushin.
  Its original **Creative Commons Attribution-NonCommercial 4.0 International**
  license is retained in `projects/unidet3d/LICENSE`. This license applies to
  these components; the root Apache license does not replace it. The released
  subset removes unrelated model/training imports and augmentation transforms,
  and validates point/superpoint lengths during evaluation sampling.
- MMEngine, MMCV, MMDetection, MMDetection3D, and MinkowskiEngine are external
  dependencies and retain their upstream licenses. This release does not vendor
  a modified MMEngine or MinkowskiEngine installation.

References:

- D. Rukhovich, A. Vorontsova, A. Konushin. *TR3D: Towards Real-Time Indoor 3D
  Object Detection*. https://arxiv.org/abs/2302.02858
- M. Kolodiazhnyi et al. *UniDet3D: Multi-dataset Indoor 3D Object Detection*.
  https://arxiv.org/abs/2409.04234
- C. Choy et al. *4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
  Networks*. https://github.com/NVIDIA/MinkowskiEngine
