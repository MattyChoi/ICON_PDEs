import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F


# Create the transformer encoder block
class SelfAttnTransformer(nn.Module):
    def __init__(
        self, 
        num_heads, 
        num_layers, 
        emb_dim, 
        widening_factor=4
    ):
        super(SelfAttnTransformer, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.widening_factor = widening_factor

        self.initializer = nn.init.xavier_uniform_

        self.ln_1 = nn.LayerNorm(emb_dim)
        self.ln_2 = nn.LayerNorm(emb_dim)

        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=emb_dim, 
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.dense_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, widening_factor * emb_dim),
                nn.GELU(),
                nn.Linear(widening_factor * emb_dim, emb_dim)
            )
            for _ in range(num_layers)
        ])


    def forward(self, x, mask=None):

        for i in range(self.num_layers):
            # First the attention block.
            attn_block = self.attn_blocks[i]
            attn, _ = attn_block(
                query=x, 
                key=x, 
                value=x, 
                key_padding_mask=mask
            )
            x = x + attn
            x = self.ln_1(x)

            # Then the dense block.
            dense_block = self.dense_blocks[i]
            dense = dense_block(x)
            x = x + dense
            x = self.ln_2(x)

        return x


class CrossAttnTransformer(nn.Module):
    def __init__(
        self, 
        num_heads, 
        num_layers, 
        emb_dim, 
        kdim, 
        vdim, 
        widening_factor=4
    ):
        super(CrossAttnTransformer, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.widening_factor = widening_factor

        self.initializer = nn.init.xavier_uniform_

        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=emb_dim, 
                num_heads=num_heads,
                kdim=kdim,
                vdim=vdim,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.dense_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, widening_factor * emb_dim),
                nn.GELU(),
                nn.Linear(widening_factor * emb_dim, emb_dim)
            )
            for _ in range(num_layers)
        ])

        self.ln_1 = nn.LayerNorm(emb_dim)
        self.ln_2 = nn.LayerNorm(emb_dim)

    def forward(self, query, key, value, mask=None, final_norm=True):

        for i in range(self.num_layers):
            # First the attention block.
            attn_block = self.attn_blocks[i]
            attn, _ = attn_block(query, key, value, key_padding_mask=mask)

            query = query + attn
            query = self.ln_1(query)

            # Then the dense block.
            dense_block = self.dense_blocks[i]
            dense = dense_block(query)

            query = query + dense
            if (i < self.num_layers - 1) and not final_norm:
                query = self.ln_2(query)

        return query


# if __name__ == "__main__":
#     query_size = 24
#     key_size = 6
#     value_size = 5
#     QK_size = 10
#     V_size = 12

#     t = 20
#     T = 40
#     query = torch.randn(t, query_size)
#     key = torch.randn(T, key_size)
#     value = torch.randn(T, value_size)

#     self_attn_transformer = SelfAttnTransformer(
#         num_heads=8,
#         num_layers=4,
#         emb_dim=query_size,
#         widening_factor=4
#     )

#     cross_attn_transformer = CrossAttnTransformer(
#         num_heads=8,
#         num_layers=4,
#         emb_dim=query_size,
#         kdim=key_size,
#         vdim=value_size,
#         widening_factor=4
#     )

#     emb = self_attn_transformer(query.unsqueeze(0))
#     out_query_cross_attn = cross_attn_transformer(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))

#     print(emb.shape)  # (20, 24)
#     print(out_query_cross_attn.shape)  # (20, 24)