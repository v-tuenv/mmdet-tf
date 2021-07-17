model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetKeras',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPNTF',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        relu_before_extra_convs=True,
        num_outs=5,
        return_funtion=True),
    bbox_head=dict(
        type='RetinaHeadSpaceSTORM',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            scale_factors=[1.,1.,1.,1.]),
            # target_means=[.0, .0, .0, .0],
            # target_stds=[1., 1., 1., 1.]),
       loss_cls=dict(
            type='FocalLossKeras',
           use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            ),
        loss_bbox=dict(type='BoxLoss')),
    # model training and testing settings
    train_cfg=Config(dict(
        assigner=dict(
            type='ArgMaxMatcher',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            iou_calculator=dict(type='IouSimilarity')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False)),
    test_cfg=Config(dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)))