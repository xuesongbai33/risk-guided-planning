_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'Bench2DriveZoo/adzoo/risk_guide_planning'

custom_imports = dict(
    imports=[
        'adzoo.risk_guide_planning.models.detectors.bevformer_risk_planner',
        'adzoo.risk_guide_planning.models.heads.riskformer_regnition_head',
        'adzoo.risk_guide_planning.models.heads.risk_guideflow_gtrs_head',
        'adzoo.risk_guide_planning.datasets.B2D_Risk_Dataset',
        'adzoo.risk_guide_planning.mmdet3d_plugin.bevformer.hooks.custom_hooks',
        'mmcv.models.dense_heads.bevformer_head',
        'mmcv.models.modules.transformer',
    ],
    allow_failed_imports=False)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

risk_class_names = [
    'car', 'truck', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'trailer', 'van', 'speed_limit',
    'traffic_light', 'stop', 'traffic_sign', 'traffic_cone', 'others'
]
bevformer_class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
]
class_names = risk_class_names

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True)

dataset_type = 'B2D_Risk_Dataset'
data_root = 'data/bench2drive'  # path to Bench2Drive raw data, replace with your own
info_root = 'data/infos'  # path to preprocessed info files, replace with your own
map_root = 'data/bench2drive/maps'  # path to map files, replace with your own
map_file = info_root + '/b2d_map_infos.pkl'
ann_file_train = info_root + '/b2d_infos_train.pkl'
ann_file_val = info_root + '/b2d_infos_val.pkl'
ann_file_test = info_root + '/b2d_infos_val.pkl'
file_client_args = dict(backend='disk')

NameMapping = {
    'vehicle.bh.crossbike': 'bicycle',
    'vehicle.diamondback.century': 'bicycle',
    'vehicle.gazelle.omafiets': 'bicycle',
    'vehicle.audi.etron': 'car',
    'vehicle.chevrolet.impala': 'car',
    'vehicle.dodge.charger_2020': 'car',
    'vehicle.dodge.charger_police': 'car',
    'vehicle.dodge.charger_police_2020': 'car',
    'vehicle.lincoln.mkz_2017': 'car',
    'vehicle.lincoln.mkz_2020': 'car',
    'vehicle.mini.cooper_s_2021': 'car',
    'vehicle.mercedes.coupe_2020': 'car',
    'vehicle.ford.mustang': 'car',
    'vehicle.nissan.patrol_2021': 'car',
    'vehicle.audi.tt': 'car',
    'vehicle.ford.crown': 'car',
    'vehicle.tesla.model3': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VWBeetle/SM_VWBeetle_parked.SM_VWBeetle_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/AudiA2/SM_AudiA2_parked.SM_AudiA2_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Chevrolet/SM_Chevrolet_parked.SM_Chevrolet_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Ford/SM_Ford_parked.SM_Ford_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mercedes/SM_Mercedes_parked.SM_Mercedes_parked': 'car',
    'vehicle.ford.ambulance': 'van',
    'vehicle.volkswagen.t2_2021': 'van',
    'vehicle.carlamotors.firetruck': 'truck',
    'vehicle.ford.raptor': 'truck',
    'vehicle.mercedes.sprinter': 'truck',
    'vehicle.tesla.cybertruck': 'truck',
    'vehicle.harley.low_rider': 'motorcycle',
    'vehicle.yamaha.yzf': 'motorcycle',
    'vehicle.kawasaki.ninja': 'motorcycle',
    'vehicle.bus.bus_0001': 'bus',
    'vehicle.bus.bus_0002': 'bus',
    'vehicle.bus.school_bus': 'bus',
    'vehicle.trailer.cargo_trailer': 'trailer',
    'vehicle.trailer.flatbed_trailer': 'trailer',
    'vehicle.trailer.car_trailer': 'trailer',
    'vehicle.construction.construction_crane': 'construction_vehicle',
    'vehicle.construction.concrete_mixer': 'construction_vehicle',
    'vehicle.construction.excavator': 'construction_vehicle',
    'vehicle.construction bulldozer': 'construction_vehicle',
    'traffic.speed_limit.30': 'speed_limit',
    'traffic.speed_limit.40': 'speed_limit',
    'traffic.speed_limit.50': 'speed_limit',
    'traffic.speed_limit.60': 'speed_limit',
    'traffic.speed_limit.90': 'speed_limit',
    'traffic.speed_limit.120': 'speed_limit',
    'traffic.speed_limit.15': 'speed_limit',
    'traffic.speed_limit.20': 'speed_limit',
    'traffic.speed_limit.25': 'speed_limit',
    'traffic.speed_limit.35': 'speed_limit',
    'traffic.speed_limit.45': 'speed_limit',
    'traffic.speed_limit.55': 'speed_limit',
    'traffic.speed_limit.65': 'speed_limit',
    'traffic.speed_limit.70': 'speed_limit',
    'traffic.speed_limit.75': 'speed_limit',
    'traffic.speed_limit.80': 'speed_limit',
    'traffic.speed_limit.85': 'speed_limit',
    'traffic.speed_limit.100': 'speed_limit',
    'traffic.speed_limit.105': 'speed_limit',
    'traffic.speed_limit.110': 'speed_limit',
    'traffic.speed_limit.115': 'speed_limit',
    'traffic.stop': 'stop',
    'traffic.yield': 'stop',
    'traffic.traffic_light': 'traffic_light',
    'traffic.speed_sign': 'traffic_sign',
    'traffic.warning_sign': 'traffic_sign',
    'traffic.no_parking_sign': 'traffic_sign',
    'traffic.no_entry_sign': 'traffic_sign',
    'static.prop.warningconstruction': 'traffic_cone',
    'static.prop.warningaccident': 'traffic_cone',
    'static.prop.trafficwarning': 'traffic_cone',
    'static.prop.constructioncone': 'traffic_cone',
    'static.prop.constructionbarrier': 'traffic_cone',
    'static.prop.barrier': 'traffic_cone',
    'walker.pedestrian.0001': 'pedestrian',
    'walker.pedestrian.0003': 'pedestrian',
    'walker.pedestrian.0004': 'pedestrian',
    'walker.pedestrian.0005': 'pedestrian',
    'walker.pedestrian.0007': 'pedestrian',
    'walker.pedestrian.0010': 'pedestrian',
    'walker.pedestrian.0013': 'pedestrian',
    'walker.pedestrian.0014': 'pedestrian',
    'walker.pedestrian.0015': 'pedestrian',
    'walker.pedestrian.0016': 'pedestrian',
    'walker.pedestrian.0017': 'pedestrian',
    'walker.pedestrian.0018': 'pedestrian',
    'walker.pedestrian.0019': 'pedestrian',
    'walker.pedestrian.0020': 'pedestrian',
    'walker.pedestrian.0021': 'pedestrian',
    'walker.pedestrian.0022': 'pedestrian',
    'walker.pedestrian.0025': 'pedestrian',
    'walker.pedestrian.0027': 'pedestrian',
    'walker.pedestrian.0030': 'pedestrian',
    'walker.pedestrian.0031': 'pedestrian',
    'walker.pedestrian.0032': 'pedestrian',
    'walker.pedestrian.0034': 'pedestrian',
    'walker.pedestrian.0035': 'pedestrian',
    'walker.pedestrian.0041': 'pedestrian',
    'walker.pedestrian.0042': 'pedestrian',
    'walker.pedestrian.0046': 'pedestrian',
    'walker.pedestrian.0047': 'pedestrian',
    'walker.pedestrian.child_0001': 'pedestrian',
    'walker.pedestrian.child_0002': 'pedestrian',
    'walker.pedestrian.child_0003': 'pedestrian',
    'walker.pedestrian.construction_0001': 'pedestrian',
    'walker.pedestrian.construction_0002': 'pedestrian',
    'walker.pedestrian.police_0001': 'pedestrian',
    'walker.pedestrian.personal_mobility_0001': 'pedestrian',
    'static.prop.dirtdebris01': 'others',
    'static.prop.dirtdebris02': 'others',
    'static.prop.rock_debris': 'others',
    'static.prop.box': 'others',
    'static.prop.container': 'others',
    'static.prop.trashcan': 'others',
    'static.prop.bench': 'others',
    'static.prop.mailbox': 'others',
    'static.prop.firehydrant': 'others',
    'static.prop.street_light': 'others',
    'static.prop.tree': 'others',
    'static.prop.vegetation': 'others',
    'static.prop.plant_pot': 'others',
    'static.prop.road_barrier': 'others',
    'static.prop.fence': 'others',
    'static.prop.wall': 'others',
    'static.prop.pole': 'others',
}

eval_cfg = {
            'dist_ths': [0.5, 1.0, 2.0, 4.0],
            'dist_th_tp': 2.0,
            'min_recall': 0.1,
            'min_precision': 0.1,
            'mean_ap_weight': 5,
            'class_names':['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian'],
            'tp_metrics':['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            'err_name_maping':{'trans_err': 'mATE','scale_err': 'mASE','orient_err': 'mAOE','vel_err': 'mAVE','attr_err': 'mAAE'},
            'class_range':{'car':(50,50),'van':(50,50),'truck':(50,50),'bicycle':(40,40),'traffic_sign':(30,30),'traffic_cone':(30,30),'traffic_light':(30,30),'pedestrian':(40,40)}
            }

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
risk_bev_h_ = 512
risk_bev_w_ = 512
high_res_risk_size = (risk_bev_h_, risk_bev_w_)
risk_threshold = 0.0
max_influence_distance = 0.8
road_centerline_dilate_m = 2.0
road_boundary_dilate_m = 0.5
road_mask_closing_m = 0.5
queue_length = 3
past_frames = 4
future_frames = 4
predict_frames = 12
planning_frames = 6
planning_vocab_path = info_root + '/planning_vocab_unified_v8192.npy'
planning_anchor_path = info_root + '/planning_anchor_unified_a1024.npy'
planning_vocab_meta_path = info_root + '/planning_vocab_unified_v8192.json'
planning_vocab_per_command = 8192
planning_anchor_per_command = 1024
planning_dynamic_candidates = 32
planning_topk = 16
samples_per_gpu = 2
workers_per_gpu = 4
prefetch_factor = 2
enable_photo_metric_distortion = True
randomize_history_queue = True

model = dict(
    _delete_=True,
    type='BEVFormerRiskPlanner',
    use_grid_mask=True,
    video_test_mode=True,
    freeze_bev=False,
    freeze_backbone=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=len(bevformer_class_names),
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='BEVFormerPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                                im2col_step=32),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.0),
                        dict(type='CustomMSDeformableAttention', embed_dims=_dim_, num_levels=1, im2col_step=32),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(bevformer_class_names)),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    risk_cognition_head=dict(
        type='RiskFormerRegnitionHead',
        in_channels=_dim_,
        risk_h=risk_bev_h_,
        risk_w=risk_bev_w_,
        bev_h=bev_h_,
        bev_w=bev_w_,
        hidden_channels=160,
        decoder_channels=128,
        highres_channels=96,
        output_mix_channels=64,
        output_mix_depth=2,
        refine_channels=64,
        future_frames=6,
        output_init_bias=-2.0),
    planning_head=dict(
        type='RiskGuideFlowGTRSHead',
        bev_channels=_dim_,
        bev_h=bev_h_,
        bev_w=bev_w_,
        command_dim=6,
        command_embed_dim=32,
        future_steps=planning_frames,
        vocab_path=planning_vocab_path,
        anchor_path=planning_anchor_path,
        vocab_meta_path=planning_vocab_meta_path,
        vocab_size=planning_vocab_per_command,
        anchor_size=planning_anchor_per_command,
        dynamic_candidates=planning_dynamic_candidates,
        static_pool_size=16,
        topk=planning_topk,
        hidden_dim=256,
        history_steps=past_frames,
        scorer_layers=3,
        vocab_dropout=0.5,
        point_cloud_range=point_cloud_range,
        generator_loss_weight=0.2,
        coarse_score_loss_weight=0.3,
        imi_score_weight=1.5,
        risk_score_weight=1.0,
        goal_progress_score_weight=1.0,
        dynamic_reward_score=0.05,
        risk_mean_weight=0.5,
        risk_max_weight=1.0,
        risk_blend_ratio=0.75,
        veto_penalty=-2.0,
        veto_penalty_far=-0.5,
        risk_perframe_loss_weight=0.05,
        risk_max_loss_weight=0.30,
        imi_sigma=5.0,
        scene_memory_latents=256,
        generator_risk_latents=256,
        scorer_risk_latents=128,
        memory_refinement_layers=1,
        num_train_anchors=8,
        veto_thresh_near=0.70,
        veto_thresh_far=0.90,
        lambda_low=0.02,
        lambda_high=0.2,
        lateral_offsets=[-1.0, 0.0, 1.0]),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))),
    test_cfg=dict(pts=dict(max_per_img=100)))

bevformer_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
    'img_norm_cfg', 'can_bus', 'folder', 'scene_token', 'sample_idx', 'frame_idx',
    'prev_bev_exists', 'world2lidar', 'ego2global', 'world2ego', 'lidar2global',
    'lidar2img', 'cam_intrinsic', 'lidar2cam')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *([dict(type='PhotoMetricDistortionMultiViewImage')] if enable_photo_metric_distortion else []),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=risk_class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=risk_class_names),
    dict(
        type='CustomCollect3D',
        keys=['img', 'gt_risk_score', 'gt_risk_valid_mask', 'sdc_planning', 'sdc_planning_mask', 'command', 'ego_his_trajs', 'ego_his_trajs_mask', 'ego_lcf_feat', 'goal_point'],
        meta_keys=bevformer_meta_keys)
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=risk_class_names, with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'gt_risk_score', 'gt_risk_valid_mask', 'sdc_planning', 'sdc_planning_mask', 'command', 'ego_his_trajs', 'ego_his_trajs_mask', 'ego_lcf_feat', 'goal_point'], meta_keys=bevformer_meta_keys)
        ])
]

data = dict(
    _delete_=True,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    prefetch_factor=prefetch_factor,
    pin_memory=True,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=risk_class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        sample_interval=5,
        name_mapping=NameMapping,
        eval_cfg=eval_cfg,
        past_frames=past_frames,
        future_frames=future_frames,
        predict_frames=predict_frames,
        planning_frames=planning_frames,
        risk_future_frames=6,
        point_cloud_range=point_cloud_range,
        high_res_risk_size=high_res_risk_size,
        risk_threshold=risk_threshold,
        max_influence_distance=max_influence_distance,
        road_centerline_dilate_m=road_centerline_dilate_m,
        road_boundary_dilate_m=road_boundary_dilate_m,
        road_mask_closing_m=road_mask_closing_m,
        randomize_history_queue=randomize_history_queue,
        map_root=map_root,
        map_file=map_file),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        classes=risk_class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        sample_interval=5,
        name_mapping=NameMapping,
        eval_cfg=eval_cfg,
        past_frames=past_frames,
        future_frames=future_frames,
        predict_frames=predict_frames,
        planning_frames=planning_frames,
        risk_future_frames=6,
        point_cloud_range=point_cloud_range,
        high_res_risk_size=high_res_risk_size,
        risk_threshold=risk_threshold,
        max_influence_distance=max_influence_distance,
        road_centerline_dilate_m=road_centerline_dilate_m,
        road_boundary_dilate_m=road_boundary_dilate_m,
        road_mask_closing_m=road_mask_closing_m,
        map_root=map_root,
        map_file=map_file),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        classes=risk_class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        sample_interval=5,
        name_mapping=NameMapping,
        eval_cfg=eval_cfg,
        past_frames=past_frames,
        future_frames=future_frames,
        predict_frames=predict_frames,
        planning_frames=planning_frames,
        risk_future_frames=6,
        point_cloud_range=point_cloud_range,
        high_res_risk_size=high_res_risk_size,
        risk_threshold=risk_threshold,
        max_influence_distance=max_influence_distance,
        road_centerline_dilate_m=road_centerline_dilate_m,
        road_boundary_dilate_m=road_boundary_dilate_m,
        road_mask_closing_m=road_mask_closing_m,
        map_root=map_root,
        map_file=map_file),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

work_dir = './work_dirs/risk_bev_guide_planning'  # directory to save logs and checkpoints, replace with your own
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=15, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0/3)
total_epochs = 16
evaluation = dict(interval=1, pipeline=test_pipeline, metric=['risk', 'planning'])
runner = dict(type='EpochBasedRunner_video', max_epochs=total_epochs,
    keys=['img', 'gt_risk_score', 'gt_risk_valid_mask', 'sdc_planning',
          'sdc_planning_mask', 'command', 'ego_his_trajs',
          'ego_his_trajs_mask', 'ego_lcf_feat', 'goal_point'])
load_from = 'ckpts/bevformer_base_b2d.pth'  # path to pretrained BEVFormer checkpoint, replace with your own
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=1)
find_unused_parameters = True
fp16 = dict(type='bfloat16')
custom_hooks = [dict(type='TransferWeight', priority='LOWEST')]
