_base_ = 'grounding_dino_owlv2.py'

# ================= 0. 变量定义 (必须定义，用于 Pipeline) =================
owlv2_model_name = '/home/chenguangyao/wzh/models/owlv2-base-patch16-ensemble'
lang_model_name = owlv2_model_name
# 指向你的 LLM 路径，用于加载 Tokenizer
lmm_path = '../huggingface/my_llava-onevision-qwen2-0.5b-ov-2/' 
lmm_max_token_length = 1200
num_region_caption = 16

# ================= 1. 修复后的 train_pipeline =================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(960, 960), keep_ratio=False), # OWLv2 专用尺寸
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # [关键补丁 1] 生成对话和区域描述数据，模型计算 Loss 必须用到
    dict(
        type='RandomSamplingNegPos2',
        tokenizer_name=lang_model_name,
        tokenizer_name2=lmm_path,
        lmm_max_token_length=lmm_max_token_length,
        num_region_caption=num_region_caption,
        num_sample_negative=85,
        max_tokens=16),
    # [关键补丁 2] 将 conversations 字段打包进 DataSample
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text', 'tags', 'contrast_conv',
                   'custom_entities', 'tokens_positive', 'dataset_mode', 
                   'conversations', 'region_conversations'))
]

model = dict(
    # ================= 2. 冻结策略 =================
    freeze_backbone=True, 
    freeze_lm=True,       
    # 仅 Projector 参与训练
)

# ================= 3. 训练循环配置 =================
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=1, 
    val_interval=1
)

# ================= 4. 数据加载器 =================
train_dataloader = dict(
    _delete_=True,
    batch_size=4, # 显存充足可改为 8
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ODVGDataset',
        data_root='../grounding_data/coco/',
        ann_file='annotations/instances_train2017_vg_merged6.jsonl',
        data_prefix=dict(img='train2017'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline, # 使用本文件中修复的 pipeline
        return_classes=True,
        actual_dataset_mode='OD',
        use_short_cap=False,
        use_uniform_prompt=True,
        clean_caption=True
    )
)

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)
