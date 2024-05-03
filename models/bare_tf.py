import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
from torch import nn, einsum

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


# Barebones Transformer
class Transformer(nn.Module):
    def __init__(
        self, 
        prompt_dim: int,
        emb_dim, 
        num_heads=8, 
        head_dim=None, 
        dropout=0.1, 
    ):
        """
        prompt_dim: number of dimensions each prompt has
        emb_dim: number of dimensions each prompt has
        num_heads: number of heads for attention mechanisms
        """
        super().__init__()

        self.emb_dim = emb_dim

        # Project the prompts to an embedding with a higher number of dimensions
        self.prompt_proj = nn.Linear(prompt_dim, emb_dim)
        super().__init__()
        
        # set the number of heads we want for multi-head attention
        self.num_heads = num_heads

        # head_dim is the new dimensions we want for each word embedding so to achieve 
        # multi-head attention, multiply this by the desired number of heads
        hidden_dim = emb_dim
        if head_dim is not None:
            hidden_dim = head_dim * num_heads

        # Project the prompts to an embedding with a higher number of dimensions
        self.prompt_proj = nn.Linear(prompt_dim, emb_dim)
            
        # create the matrices needed to compute the query, key, and value vectors
        self.c_attn = nn.Linear(emb_dim, hidden_dim * 3)

        # one last mlp to combine all information from all the heads
        self.c_proj = nn.Linear(hidden_dim, prompt_dim)

        # dropouts for qkv computing and multi-head attention projection
        self.dropout = nn.Dropout(dropout)
        self.mha_dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, prompt):
        # project the prompts to a higher embedding space
        x = self.prompt_proj(prompt)
        
        # batch, position (timestep), dimension of embedding
        b, t, c = x.shape

        # get the query, value, and key vectors
        qkv = self.c_attn(x).chunk(3, dim = 2)

        # rearrange the vectors so that we separate by each head
        # new shape = batch, head, position(timestep), head_dim
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h = self.num_heads), qkv)

        # matrix multiply the query and key vectors to get a table of the dot products
        # of each query and key vector at every pair of timesteps in the given block
        # i is the dimension for the query vector, and k is the dimenstion for the key vector
        # scale this product by the square root of the dimension of the vectors
        scores = einsum('b h i d, b h j d -> b h i j', q, k) / (k.size(-1) ** 0.5)

        # softmax over the last dimension (the dimension of the key vectors)
        attn = scores.softmax(dim = -1)
        attn = self.dropout(attn)

        # basically a linear combination where attn is the weights and v is the values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # concatenate all the heads now
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.mha_dropout(self.c_proj(out))
        
        return out
    