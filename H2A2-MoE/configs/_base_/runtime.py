default_scope = 'mmdet3d'
custom_imports = dict(
    imports=['projects.TR3D.tr3d', 'projects.unidet3d.unidet3d'],
    allow_failed_imports=False)
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=None,
    checkpoint=None,
    sampler_seed=None)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=False)
log_level = 'INFO'
randomness = dict(seed=None, deterministic=False)
visualizer = dict(type='Visualizer', vis_backends=[])
