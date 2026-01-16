# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoTokenizer
    from transformers import Owlv2Model as HFOwlv2Model
    from transformers import Owlv2TextModel as HFOwlv2TextModel
except ImportError:
    AutoTokenizer = None
    HFOwlv2Model = None
    HFOwlv2TextModel = None

from mmdet.registry import MODELS


def _build_text_model(name: str, use_checkpoint: bool):
    if HFOwlv2TextModel is not None:
        model = HFOwlv2TextModel.from_pretrained(name)
    elif HFOwlv2Model is not None:
        model = HFOwlv2Model.from_pretrained(name).text_model
    else:
        raise RuntimeError(
            'transformers is not installed, please install it by: '
            'pip install transformers.')

    if use_checkpoint and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return model


def _get_hidden_size(model) -> int:
    cfg = getattr(model, 'config', None)
    if cfg is not None:
        if hasattr(cfg, 'hidden_size'):
            return cfg.hidden_size
        if hasattr(cfg, 'text_config') and hasattr(cfg.text_config, 'hidden_size'):
            return cfg.text_config.hidden_size
    if hasattr(model, 'text_model') and hasattr(model.text_model, 'config'):
        return model.text_model.config.hidden_size
    raise RuntimeError('Failed to infer text hidden size for OWLv2 text model.')


class Owlv2TextEncoder(nn.Module):
    """OWLv2 text encoder wrapper."""

    def __init__(self, name: str, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.model = _build_text_model(name, use_checkpoint)
        self.language_dim = _get_hidden_size(self.model)

    def forward(self, x) -> dict:
        mask = x['attention_mask']
        outputs = self.model(
            input_ids=x['input_ids'],
            attention_mask=mask,
            output_hidden_states=True,
        )
        features = outputs.last_hidden_state
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float().to(features.dtype)
        else:
            embedded = features

        text_mask = mask.bool()
        if text_mask.dim() == 2:
            text_self_attention_mask = text_mask[:, None, :] & text_mask[:, :, None]
        else:
            text_self_attention_mask = text_mask

        hidden = outputs.hidden_states[-1] if outputs.hidden_states else features
        return {
            'embedded': embedded,
            'masks': text_self_attention_mask,
            'hidden': hidden,
        }


@MODELS.register_module()
class Owlv2TextModel(BaseModel):
    """OWLv2 text model for language embedding only encoder."""

    def __init__(self,
                 name: str = 'google/owlv2-base-patch16-ensemble',
                 max_tokens: int = None,
                 pad_to_max: bool = True,
                 use_fast: bool = True,
                 use_checkpoint: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
        self.max_tokens = max_tokens or self.tokenizer.model_max_length
        self.pad_to_max = pad_to_max

        self.language_backbone = nn.Sequential(
            OrderedDict([('body', Owlv2TextEncoder(name, use_checkpoint=use_checkpoint))]))

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_tensors='pt',
            truncation=True).to(device)

        tokenizer_input = {
            'input_ids': tokenized.input_ids,
            'attention_mask': tokenized.attention_mask,
        }
        language_dict_features = self.language_backbone(tokenizer_input)
        language_dict_features['position_ids'] = torch.arange(
            tokenized.input_ids.shape[1],
            device=tokenized.input_ids.device).unsqueeze(0).repeat(
                tokenized.input_ids.shape[0], 1)
        language_dict_features['text_token_mask'] = tokenized.attention_mask.bool()
        return language_dict_features
