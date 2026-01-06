""" Multi Step Attention for CNN """

import torch
import torch.nn as nn
import torch.nn.functional as F


SCALE_WEIGHT = 0.5**0.5


def seq_linear(linear, x):
    """
    Apply linear to Conv1d-style tensor (B, C, T)
    Returns same shape (B, C, T)
    """
    batch, channels, length = x.size()
    # transpose to (B, T, C) to apply linear
    h = linear(x.transpose(1, 2).contiguous().view(batch * length, channels))
    h = h.view(batch, length, channels).transpose(1, 2)  # back to (B, C, T)
    return h


class ConvMultiStepAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_in = nn.Linear(input_size, input_size)
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, base_target_emb, input_from_dec, encoder_out_top, encoder_out_combine):
        """
        base_target_emb: (B, C, T)
        input_from_dec: (B, C, T)
        encoder_out_top: (B, T_enc, C)
        encoder_out_combine: (B, T_enc, C)
        """
        # linear over channels
        preatt = seq_linear(self.linear_in, input_from_dec)  # (B, C, T)
        target = (base_target_emb + preatt) * SCALE_WEIGHT  # (B, C, T)

        # compute attention: (B, T_dec, C) x (B, C, T_enc) -> (B, T_dec, T_enc)
        target_t = target.transpose(1, 2)  # (B, T_dec, C)
        pre_attn = torch.bmm(target_t, encoder_out_top.transpose(1, 2))  # (B, T_dec, T_enc)

        if self.mask is not None:
            pre_attn.data.masked_fill_(self.mask, -float("inf"))

        attn = F.softmax(pre_attn, dim=2)  # (B, T_dec, T_enc)

        # compute context: weighted sum of encoder outputs
        context_output = torch.bmm(attn, encoder_out_combine)  # (B, T_dec, C)
        context_output = context_output.transpose(1, 2)  # back to (B, C, T_dec)
        return context_output, attn
