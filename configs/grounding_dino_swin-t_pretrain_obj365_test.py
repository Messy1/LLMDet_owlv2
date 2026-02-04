_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

dataset_type = 'LVISV1Dataset'
data_root = '../grounding_data/coco/'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
        return_classes=True))

test_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival_inserted_image_name.json')

default_hooks = dict(
    # 覆盖继承自 _base_ 的 checkpoint 设置
    checkpoint=dict(
        type='CheckpointHook', 
        interval=500,    # 每 2000 次迭代存一次，不再是 25 小时存一次了
        by_epoch=False,   # 必须显式设为 False，interval 才会按 Iter 计算
        max_keep_ckpts=3, # 建议只保留最近 3 个，Objects365 的 checkpoint 很大
        save_last=True    # 始终保留最新的那个，方便 --resume
    ),
    # 顺便把日志打印调快一点，方便你在终端盯着
    logger=dict(type='LoggerHook', interval=10) 
)

env_cfg = dict(
    dist_cfg=dict(
        backend='nccl', 
        timeout=28800 # 8 Hours
    )
)