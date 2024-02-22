import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F


# In-Context Operator Network
class ICON(nn.Module):
    def __init__(
        self, 
        max_length: int, 
        emb_dim: int,
        num_heads: int, 
        num_enc_layers: int,
        num_dec_layers: int,
        q_size: int,
        kv_size: int,
        qoi_v_size: int,
        QK_size: int,
        V_size: int,
        initializer: str = 'glorot_uniform',
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
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers


    def forward(self, prompt, query, labels):
        """
        prompt: outputs of a string of words through a tokenizer, should have shape (batch_size, emb_dim, max_length)
        attn_mask: vector of the same shape as idx with 0s for pad tokens and 0s for the rest
        targets: same as idx, should have shape (batch_size, vocab_size)
        """
        # get the device the inputs are being trained on
        device = idx.device

        # b is the batch size, t is the number of tokens
        b, t = idx.shape

        assert t <= self.max_length, f"Cannot forward sequence of length {t}, block size is only {self.max_length}"

        pos = torch.arange(t, dtype=torch.long, device=device)

        # get the word and positional embeddings
        tok_emb = self.transformer.wte(idx) # shape is (b, t, emb_dim)
        pos_emb = self.transformer.wpe(pos) # shape is (t, c)

        # put through dropout
        x = self.transformer.dropout(tok_emb + pos_emb) # shape is (b, t, emb_dim)

        # create the attention mask for the causal attention mechanism
        if attn_mask is not None:
            attn_mask = attn_mask.view(b, -1)       # make sure it's the same shape as the tokens
            attn_mask = attn_mask[:, None, None, :]
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min

        # put it through the decoder blocks
        for block in self.transformer.h:
            x = block(x, attn_mask) # shape is (b, t, emb_dim)

        # apply the last layer norm
        x = self.transformer.ln_f(x) # shape is (b, t, emb_dim)

        loss = None
        # get the scores for each vocab
        logits = self.lm_head(x) # shape is (b, t, vocab_size)
        if labels is not None:

            # shift the logits and labels so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # combine the batch and timestep axes for better parallelization
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
            )
        
        return logits, loss