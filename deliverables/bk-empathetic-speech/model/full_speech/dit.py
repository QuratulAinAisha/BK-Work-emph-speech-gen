"""Non-autoregressive conditional DiT. / 비자기회귀 조건부 DiT."""

import torch
from torch import nn

from .tensor_ops import sinusoidal


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, memory, global_condition, mask, memory_mask):
        parameters = self.modulation(global_condition).chunk(9, dim=-1)
        for index, operation in enumerate((self.self_attention, self.cross_attention, self.mlp)):
            shift, scale, gate = parameters[index * 3:index * 3 + 3]
            query = self.norms[index](x) * (1 + scale[:, None]) + shift[:, None]
            if index == 0:
                update = operation(query, query, query, key_padding_mask=~mask, need_weights=False)[0]
            elif index == 1:
                update = operation(query, memory, memory, key_padding_mask=~memory_mask,
                                   need_weights=False)[0]
            else:
                update = operation(query)
            x = (x + gate[:, None].tanh() * update).masked_fill(~mask[..., None], 0)
        return x


class ConditionalDiT(nn.Module):
    def __init__(self, input_dim, local_dim, config, layers):
        super().__init__()
        dim = config.hidden_dim
        self.dim = dim
        self.input_projection = nn.Linear(input_dim, dim)
        self.local_projection = nn.Linear(local_dim, dim)
        self.context_projection = nn.Linear(512, dim)
        self.time_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.length_projection = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, config.num_heads, config.dropout) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, input_dim))

    def forward(self, x, time, local_condition, context, global_condition, mask, context_mask):
        # Time, length, style and speaker enter every block. / 시간·길이·스타일·화자가 각 층에 들어갑니다.
        timing = self.time_projection(sinusoidal(time * 1000, self.dim).to(x.dtype))
        length = mask.sum(1).float().log()[:, None].to(x.dtype)
        condition = global_condition + timing + self.length_projection(length)
        positions = sinusoidal(torch.arange(x.shape[1], device=x.device), self.dim).to(x.dtype)
        hidden = self.input_projection(x) + self.local_projection(local_condition) + positions[None]
        hidden = hidden.masked_fill(~mask[..., None], 0)
        memory = self.context_projection(context.masked_fill(~context_mask[..., None], 0))
        for block in self.blocks:
            hidden = block(hidden, memory, condition, mask, context_mask)
        return self.output(hidden).masked_fill(~mask[..., None], 0)
