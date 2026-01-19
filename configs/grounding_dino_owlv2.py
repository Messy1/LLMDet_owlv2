_base_ = 'grounding_dino_swin_t.py'

owlv2_model_name = '/home/chenguangyao/wzh/models/owlv2-base-patch16-ensemble'

# ================= 1. 显式定义 Pipeline (确保 CLIP/OWLv2 兼容) =================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(960, 960), keep_ratio=False), # OWLv2 锁死 960
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction', 'text', 
                    'custom_entities', 'tokens_positive', 'dataset_mode'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 960), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities'))
]

model = dict(
    # ================= 2. 核心逻辑控制 =================
    use_p4_input=False,  # 关闭 P4
    use_p5_input=True,   # 只用 P5 (单尺度)
    
    # ================= 3. 数据预处理 (使用 CLIP 标准) =================
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[122.77, 116.75, 104.09], # CLIP 均值
        std=[68.5, 66.6, 70.39],       # CLIP 标准差
        bgr_to_rgb=True,
        pad_mask=False,
    ),

    # ================= 4. Backbone 替换 =================
    backbone=dict(
        _delete_=True, # 删除 Swin 配置
        type='OWLv2Backbone',
        model_name=owlv2_model_name,
        freeze=True
    ),

    # ================= 4.1 文本编码器替换 (OWLv2 Text) =================
    language_model=dict(
        _delete_=True,
        type='Owlv2TextModel',
        name=owlv2_model_name,
        max_tokens=16,
        pad_to_max=True,
        use_fast=True,
    ),

    # ================= 5. Neck 调整 (单尺度输入输出) =================
    neck=dict(
        type='ChannelMapper',
        in_channels=[768], # OWLv2-Base 输出维度
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1  # 关键：只输出一层给 Transformer
    ),

    # ================= 6. Transformer 结构同步 (Deformable Attn 改为单层) =================
    encoder=dict(
        layer_cfg=dict(
            self_attn_cfg=dict(num_levels=1), # 必须设为 1
        )
    ),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(num_levels=1), # 必须设为 1
        )
    ),
    
    # 移除 bbox_head 中的 num_query 错误
    bbox_head=dict(
        # 继承 _base_ 中的 num_classes=256 等参数，不再额外添加报错参数
        contrastive_cfg=dict(max_text_len=16),
    ),
)

# 确保 Dataloader 使用正确的 Pipeline
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
# 这样修改既能解决 ConcatDataset 的报错，又能满足“快速试试”的要求
train_dataloader = dict(
    batch_size=2, # 每张卡 2 个样本，8 张卡就是 16
    num_workers=4,
    dataset=dict(
        _delete_=True, # 彻底删除基类的 ConcatDataset 结构
        type='ODVGDataset', # 显式指定数据集类型
        data_root='../grounding_data/coco/', # 数据集根目录
        ann_file='annotations/instances_train2017_vg_merged6.jsonl', # 使用合并后的注释文件
        data_prefix=dict(img='train2017'),
        pipeline=train_pipeline, # 使用适配 OWLv2 的 960x960 pipeline
        return_classes=True,
        actual_dataset_mode='OD'))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
