import os
import sys
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .custom_op import FakeLinear
from .spinner import spinner_run
from .torch_utils import onnx_export
from .transformers import repeat_kv, rotate_half


def build_target_layer_ids(num_target_layers, num_draft_layers):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


class DFlashAttention(nn.Module):
    def __init__(self, attn, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm

    @staticmethod
    def apply_rotary_pos(query_states, key_states, cos, sin):
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            cos = cos.transpose(1, 2)
            sin = sin.transpose(1, 2)
        query_len = query_states.size(-2)
        query_states = (query_states * cos[..., -query_len:, :]) + (rotate_half(query_states) * sin[..., -query_len:, :])
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        return query_states, key_states

    def forward(self, hidden_states, target_hidden, rotary_pos_emb, attention_mask, past_key_value=None):
        bsz, query_len, _ = hidden_states.size()
        ctx_len = target_hidden.shape[1]

        query_states = self.q_proj(hidden_states)
        key_ctx = self.k_proj(target_hidden)
        key_noise = self.k_proj(hidden_states)
        value_ctx = self.v_proj(target_hidden)
        value_noise = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, query_len, self.num_attention_heads, self.head_dim)
        key_states = torch.cat([key_ctx, key_noise], dim=1).view(
            bsz, ctx_len + query_len, self.num_key_value_heads, self.head_dim)
        value_states = torch.cat([value_ctx, value_noise], dim=1).view(
            bsz, ctx_len + query_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states)
        cos, sin = rotary_pos_emb[0], rotary_pos_emb[1]
        query_states, key_states_t = self.apply_rotary_pos(query_states, key_states.transpose(1, 2), cos, sin)
        key_states = key_states_t.transpose(1, 2)

        if past_key_value is not None:
            past_key, past_value = past_key_value[0], past_key_value[1]
            key_states = torch.cat((past_key, key_states), dim=1)
            value_states = torch.cat((past_value, value_states), dim=1)
        present_key_value = torch.stack((key_states, value_states))

        key_states = key_states.permute([0, 2, 3, 1])
        value_states = value_states.transpose(1, 2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states) / (self.head_dim ** 0.5)
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, query_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value


class DFlashLayer(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.self_attn = DFlashAttention(layer.self_attn, config)
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm

    def forward(self, hidden_states, target_hidden, rotary_pos_emb, attention_mask, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, target_hidden, rotary_pos_emb, attention_mask, past_key_value)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class DFlash(nn.Module):
    def __init__(self, dflash_path, base):
        super().__init__()
        self.dflash_path = dflash_path
        if dflash_path not in sys.path:
            sys.path.insert(0, dflash_path)
        config = AutoConfig.from_pretrained(dflash_path, trust_remote_code=True)
        self.draft = AutoModel.from_pretrained(dflash_path, trust_remote_code=True).eval()
        self.config = config
        self.lm = nn.Linear(
            base.lm.lm.in_features,
            base.lm.lm.out_features,
            base.lm.lm.bias is not None,
        )
        self.lm.weight = base.lm.lm.weight
        if base.lm.lm.bias is not None:
            self.lm.bias = base.lm.lm.bias
        self.hidden_size = config.hidden_size
        self.block_size = config.block_size
        self.mask_token_id = config.dflash_config.get("mask_token_id", None)
        self.target_layer_ids = config.dflash_config.get("target_layer_ids", None)
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.layers = nn.ModuleList([DFlashLayer(layer, config) for layer in self.draft.layers])
        self.rotary = self.draft.rotary_emb
        self.fc = self.draft.fc
        self.hidden_norm = self.draft.hidden_norm
        self.norm = self.draft.norm
        self.unloaded_ops = {}

    def unload_param(self):
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                for name, child in layer.self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(layer.self_attn, name, build_faker(child, f'/dflash_layers.{i}/self_attn/{name}/Linear'))
                for name, child in layer.mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(layer.mlp, name, build_faker(child, f'/dflash_layers.{i}/mlp/{name}/Linear'))
            self.fc = build_faker(self.fc, '/dflash/fc/Linear')
            self.lm = build_faker(self.lm, '/dflash/lm_head/Linear')

    @spinner_run(f'export onnx model to ')
    def export(self, onnx_path):
        self.unload_param()
        onnx_model = f'{onnx_path}/dflash.onnx'
        seq_len = self.block_size
        target_len = 1
        noise_embedding = torch.ones([seq_len, 1, self.hidden_size], dtype=torch.float)
        target_hidden = torch.ones([target_len, 1, len(self.target_layer_ids) * self.hidden_size], dtype=torch.float)
        attention_mask = torch.zeros([1, 1, seq_len, target_len + seq_len], dtype=torch.float)
        position_ids = torch.arange(target_len + seq_len, dtype=torch.int).unsqueeze(0)

        with torch.no_grad():
            onnx_export(
                self,
                (noise_embedding, target_hidden, attention_mask, position_ids),
                onnx_model,
                input_names=[
                    'noise_embedding', 'target_hidden', 'attention_mask',
                    'position_ids'
                ],
                output_names=['logits', 'presents'],
                dynamic_axes={
                    'noise_embedding': {0: 'seq_len'},
                    'target_hidden': {0: 'target_len'},
                    'attention_mask': {2: 'seq_len', 3: 'kv_len'},
                    'position_ids': {1: 'kv_len'},
                })
        return onnx_model

    def forward(self, noise_embedding, target_hidden, attention_mask, position_ids):
        noise_embedding = noise_embedding.permute(1, 0, 2)
        target_hidden = target_hidden.permute(1, 0, 2)
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        rotary_input = torch.cat([target_hidden, hidden_states], dim=1)
        rotary_pos_emb = self.rotary(rotary_input, position_ids)
        presents = []
        for i, layer in enumerate(self.layers):
            hidden_states, present_key_value = layer(
                hidden_states, target_hidden, rotary_pos_emb, attention_mask)
            presents.append(present_key_value)
        hidden_states = self.norm(hidden_states)
        logits = self.lm(hidden_states).permute(1, 0, 2)
        return logits, torch.stack(presents)
