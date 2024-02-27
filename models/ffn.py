import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn


# Feed Forward Netwrok
class FFN(nn.Module):
    def __init__(
        self, 
        max_length: int, 
        prompt_dim: int,
        qoi_size: int,
        emb_dim: int,
        num_layers: int,
        widening_factor=4,
    ):
        """
        max_length: the maximum length of each prompt
        emb_dim: number of dimensions each prompt has
        qoi_size: dim for qoi_v after postprocess projection
        num_heads: number of heads for attention mechanisms
        num_layers: number of encoder blocks
        """
        super().__init__()

        self.max_length = max_length
        self.emb_dim = emb_dim

        # Project the prompts to an embedding with a higher number of dimensions
        self.prompt_proj = nn.Linear(prompt_dim, emb_dim)

        self.dense_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, widening_factor * emb_dim),
                nn.GELU(),
                nn.Linear(widening_factor * emb_dim, emb_dim)
            )
            for _ in range(num_layers)
        ])

        # Project the encoder output to the output space
        self.qoi_proj = nn.Linear(emb_dim, qoi_size)
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, prompt, query):
        """
        prompt: 2D array, [batch_size, prompt_size, prompt_dim (number of keys, values, inds)]
        """
        prompt = prompt[:, :, [1, 2]]
        
        # project the prompts to a higher embedding space
        x = self.prompt_proj(prompt)

        for blocks in self.dense_blocks:
            x = blocks(x)

        # infer the qoi from the decoder embeddings
        qoi = self.qoi_proj(x).squeeze(-1)

        return qoi
