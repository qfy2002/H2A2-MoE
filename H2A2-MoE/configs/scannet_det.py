# Evaluation configuration. Dataset paths are relative to this repository.
_base_ = [
    './_base_/runtime.py',
]
data_root = 'data/scannet_det/'
model = dict(
    backbone=dict(
        block_conv_cfg=dict(num_private_experts=1, num_shared_experts=3),
        block_conv_type='multi',
        depth=34,
        in_channels=3,
        norm='batch',
        num_planes=(
            64,
            128,
            128,
            128,
        ),
        type='TR3DMinkResNet'),
    bbox_head=dict(
        in_channels=128,
        label2level=[
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ],
        num_reg_outs=6,
        pts_center_threshold=6,
        type='TR3DHead',
        voxel_size=0.01),
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    neck=dict(
        in_channels=(
            64,
            128,
            128,
            128,
        ), out_channels=128, type='TR3DNeck'),
    test_cfg=dict(iou_thr=0.5, nms_pre=1000, score_thr=0.01),
    type='MinkSingleStage3DDetector')
test_pipeline = [
    dict(
        coord_type='DEPTH',
        load_dim=6,
        shift_height=False,
        type='LoadPointsFromFile',
        use_color=True,
        use_dim=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]),
    dict(rotation_axis=2, type='GlobalAlignment'),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(color_mean=None, type='NormalizePointsColor'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='scannet_infos_val.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root=data_root,
        metainfo=dict(
            classes=(
                'cabinet',
                'bed',
                'chair',
                'sofa',
                'table',
                'door',
                'window',
                'bookshelf',
                'picture',
                'counter',
                'desk',
                'curtain',
                'refrigerator',
                'showercurtrain',
                'toilet',
                'sink',
                'bathtub',
                'garbagebin',
            )),
        pipeline=test_pipeline,
        test_mode=True,
        type='ScanNetDataset'),
    num_workers=1,
    sampler=dict(_scope_='mmdet3d', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='IndoorMetric')
