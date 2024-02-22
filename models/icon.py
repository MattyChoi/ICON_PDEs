import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import SelfAttnTransformer, CrossAttnTransformer


# In-Context Operator Network
class ICON(nn.Module):
    def __init__(
        self, 
        max_length: int, 
        prompt_dim: int,
        query_dim: int,
        qoi_size: int,
        emb_dim: int,
        num_heads: int, 
        num_enc_layers: int,
        num_dec_layers: int,
        widening_factor: int = 4
    ):
        """
        max_length: the maximum length of each prompt
        emb_dim: number of dimensions each prompt has
        num_heads: number of heads for attention mechanisms
        num_enc_layers: number of encoder blocks
        num_dec_layers: number of decoder blocks
        q_size: dim for query after preprocess projection
        kv_size: dim for key and value after preprocess projection
        qoi_v_size: dim for qoi_v after postprocess projection
        QK_size: dim for Q and K in self-attn
        V_size: dim for V in attn
        """
        super().__init__()

        self.max_length = max_length
        self.emb_dim = emb_dim

        # Project the prompts to an embedding with a higher number of dimensions
        self.prompt_proj = nn.Linear(prompt_dim, emb_dim)

        # Project the query keys to an embedding with a higher number of dimensions
        self.q_proj = nn.Linear(query_dim, emb_dim)

        # create the encoder part of the transformer
        self.encoder = SelfAttnTransformer(
            num_heads=num_heads,
            num_layers=num_enc_layers,
            emb_dim=emb_dim,
            widening_factor=widening_factor,
        )   

        # create the decoder part of the transformer
        self.decoder = CrossAttnTransformer(
            num_heads=num_heads,
            num_layers=num_dec_layers,
            emb_dim=emb_dim,
            kdim=emb_dim,
            vdim=emb_dim,
            widening_factor=widening_factor,
        )

        # Project the decoder output to the output space
        self.qoi_proj = nn.Linear(emb_dim, qoi_size)


    def forward(self, prompt, query):
        """
        prompt: 2D array, [batch_size, propmt_size, prompt_dim (number of keys, values, inds)]
        query: 2D array, query representing the key of qoi, [batch_size, qoi_size, query_dim]
        """

        # project the prompts to a higher embedding space
        prompt = self.prompt_proj(prompt)

        # get the encoder embeddings of the prompts
        enc = self.encoder(prompt)

        # project the query keys to a higher embedding space
        query = self.q_proj(query)

        # get the decoder embeddings of the query
        dec = self.decoder(query=query, key=enc, value=enc)

        # infer the qoi from the decoder embeddings
        qoi = self.qoi_proj(dec).squeeze(-1)

        return qoi