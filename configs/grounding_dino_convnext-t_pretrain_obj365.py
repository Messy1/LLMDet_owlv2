_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

# -----------------------------------------------------------------------------
# 1. 模型架构修改 (Model Architecture)
# -----------------------------------------------------------------------------
model = dict(
    # 彻底替换 Swin 为 ConvNeXt-V2
    backbone=dict(
        _delete_=True,
        type='ConvNeXt',
        out_indices=(1, 2, 3), # 对应你探测出的 P3(192), P4(384), P5(768)
        gap_before_final_norm=False,
        use_grn=True,
        layer_scale_init_value=0.,
        drop_path_rate=0.2,
        # 设为 None，因为所有权重都从底部的 load_from 中一次性加载
        init_cfg=None
    ),
    # 确保 Neck (ChannelMapper) 接收正确的通道数
    neck=dict(
        in_channels=[192, 384, 768], # 与 ConvNeXt 输出严格对齐
        out_channels=256,
        num_outs=4,
        # 权重列表里没有 neck. 开头的 Key，所以 Neck 会根据 init_cfg 自动随机初始化
        init_cfg=dict(type='Xavier', layer='Conv2d')
    ),
    # 保持 Transformer Encoder (Feature Enhancer) 设置不变
    # 它将加载缝合后的旧权重，开始适应新的卷积特征
    test_cfg=dict(max_per_img=300, chunked_size=40,)
)

# -----------------------------------------------------------------------------
# 2. 权重加载 (Weight Loading)
# -----------------------------------------------------------------------------
# 指向你缝合好的“终极版”权重文件
load_from = '/home/chenguangyao/wzh/workspace/LLMDetProj/huggingface/mmgdino_convnextv2_tiny.pth'

# -----------------------------------------------------------------------------
# 3. 训练策略与学习率 (Training Strategy & LR)
# -----------------------------------------------------------------------------
# 线性调整说明：
# 原始参考：BS=128, LR=0.0001
# 你的环境：BS=32 (8卡 x 4bs), 比例为 0.25 -> 基础 LR 应为 0.000025
# 但由于 Neck 和 Encoder 需要较多磨合，我们适当微调
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.000028125,  # 针对 6 卡 BS=36 线性缩放后的学习率
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # 1. 冻结 BERT 语言模型 (lr_mult=0.0)，防止视觉端噪声污染成熟的语言表征
            'language_model': dict(lr_mult=0.0),
            
            # 2. 视觉主干 (Backbone) 给小学习率 (0.1倍)，让其缓慢适应检测任务，保护 ImageNet 预训练知识
            'backbone': dict(lr_mult=0.1),
            
            # 3. Neck 是随机初始化的“新手”，Encoder 是需要“再教育”的“熟手”
            # 给标准学习率 (1.0倍) 让它们全速对齐特征
            'neck': dict(lr_mult=1.0),
            'encoder': dict(lr_mult=1.0),
            
            # 4. Decoder 和 Head 已经有检测经验，正常训练即可
            'decoder': dict(lr_mult=1.0),
            'bbox_head': dict(lr_mult=1.0),
        }))

# -----------------------------------------------------------------------------
# 4. 调度器设置 (Learning Policy)
# -----------------------------------------------------------------------------
max_epochs = 30
param_scheduler = [
    # 延长 Warmup 到 2000 个 Iter，给随机初始化的 Neck 足够的缓冲时间
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 26],
        gamma=0.1)
]

# -----------------------------------------------------------------------------
# 5. 其他运行设置 (Runtime)
# -----------------------------------------------------------------------------
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

train_dataloader = dict(
    batch_size=6,  # 每张卡 4 个样本
    num_workers=12
)

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

# 告诉 MMEngine 基于当前的 BS=32 自动缩放（如果以后你改变了每张卡的 BS）
auto_scale_lr = dict(base_batch_size=24)

# 开启混合精度训练 (AMP)，充分压榨 A800 的性能
# ./tools/dist_train.sh <config> 8 --amp

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