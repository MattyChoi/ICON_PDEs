import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn

from models.transformer import SelfAttnTransformer


# In-Context Operator Network Encoder
class Encoder(nn.Module):
    def __init__(
        self, 
        max_length: int, 
        prompt_dim: int,
        qoi_size: int,
        emb_dim: int,
        num_heads: int, 
        num_layers: int,
        widening_factor: int = 4
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

        # create the encoder part of the transformer
        self.encoder = SelfAttnTransformer(
            num_heads=num_heads,
            num_layers=num_layers,
            emb_dim=emb_dim,
            widening_factor=widening_factor,
        )   

        # Project the encoder output to the output space
        self.qoi_proj = nn.Linear(emb_dim, qoi_size)
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, prompt, query):
        """
        prompt: 2D array, [batch_size, propmt_size, prompt_dim (number of keys, values, inds)]
        """

        # project the prompts to a higher embedding space
        prompt = self.prompt_proj(prompt)

        # get the encoder embeddings of the prompts
        enc = self.encoder(prompt)

        # infer the qoi from the decoder embeddings
        qoi = self.qoi_proj(enc).squeeze(-1)

        return qoi