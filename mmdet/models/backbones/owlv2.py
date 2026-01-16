import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from transformers import Owlv2Model

@MODELS.register_module()
class OWLv2Backbone(BaseModule):
    """
    OWLv2 Backbone for LLMDet.
    输出: 单尺度特征图 (Tuple with 1 tensor).
    """
    def __init__(self, 
                 model_name="google/owlv2-base-patch16-ensemble",
                 freeze=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        print(f"Loading OWLv2 Backbone from: {model_name} ...")
        # 加载 Hugging Face 预训练模型
        self.owl_model = Owlv2Model.from_pretrained(model_name)
        self.vision_model = self.owl_model.vision_model
        
        # 冻结参数 (强烈建议，防止破坏预训练特征)
        if freeze:
            self.vision_model.eval()
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        # x shape: [Batch, 3, H, W]
        # 注意：输入必须已经经过了 Resize (例如 960x960) 和 Normalize
        
        # 1. OWLv2 推理
        outputs = self.vision_model(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state # [B, 3601, 768]
        
        # 2. 去掉 CLS Token (第0个)
        features = last_hidden_state[:, 1:, :] # [B, 3600, 768]
        
        # 3. Reshape 回 2D 图片格式 [B, C, H, W]
        B, L, C = features.shape
        H = W = int(L**0.5) # 确保 L 是完全平方数 (60*60=3600)
        
        # [B, L, C] -> [B, C, L] -> [B, C, H, W]
        features = features.permute(0, 2, 1).view(B, C, H, W)
        
        # 4. 返回 Tuple (MMDet 要求 backbone 返回 tuple)
        # 我们只返回这一层，作为 "P5"
        return (features,)