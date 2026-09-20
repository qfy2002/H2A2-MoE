"""Register model and evaluation components without training extensions."""
from .axis_aligned_iou_loss import TR3DAxisAlignedIoULoss
from .rotated_iou_loss import TR3DRotatedIoU3DLoss
from .mink_resnet import TR3DMinkResNet
from .minkowski_conv import TR3DMinkowskiConvolutionMulti
from .tr3d_head import TR3DHead
from .tr3d_neck import TR3DNeck
from .tr3d_seg import (TR3DOfficialMinkowskiPreprocessor, TR3DMinkSegResNet,
                       TR3DMinkUNet, TR3DMinkUNetDecoder, TR3DMinkUNetHead)
from .transforms_3d import (TR3DKeepPointsForVisualization,
                            TR3DOfficialSparseQuantize, TR3DPack3DDetInputs)
from .detection_vis_metric import TR3DDetectionVisMetric
from .seg_vis_metric import (TR3DSegmentationVisMetric,
                             TR3DSegmentationRankingMetric)
