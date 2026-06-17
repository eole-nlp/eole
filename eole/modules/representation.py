import torch
import torch.nn as nn


class LayerwiseAttention(nn.Module):
    def __init__(self, num_layers, layer_transformation="softmax", layer_norm=False):
        super().__init__()
        self.transform_fn = torch.softmax
        self.layer_norm = layer_norm
        if layer_transformation == "sparsemax":
            from entmax import sparsemax

            self.transform_fn = sparsemax
        self.scalar_parameters = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True) for _ in range(num_layers)]
        )
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, tensors, mask=None):
        weights = torch.cat([parameter for parameter in self.scalar_parameters])
        normed_weights = self.transform_fn(weights, dim=0)
        chunks = torch.split(normed_weights, 1)
        if not self.layer_norm:
            return self.gamma * sum(w * t for w, t in zip(chunks, tensors))

        if mask is None:
            raise ValueError("mask is required when layer_norm=True for layerwise attention")

        def _layer_norm(tensor, broadcast_mask, mask_float):
            tensor_masked = tensor * broadcast_mask
            batch_size, _, input_dim = tensor.size()
            num_elements_not_masked = mask_float.sum(1) * input_dim
            mean = tensor_masked.view(batch_size, -1).sum(1)
            mean = (mean / num_elements_not_masked).view(batch_size, 1, 1)
            variance = (((tensor_masked - mean) * broadcast_mask) ** 2).view(batch_size, -1).sum(1)
            variance = variance / num_elements_not_masked
            normalized = (tensor - mean) / torch.sqrt(variance + 1e-12).view(batch_size, 1, 1)
            return normalized

        mask_float = mask.float()
        broadcast_mask = mask_float.unsqueeze(-1)
        pieces = []
        for weight, tensor in zip(chunks, tensors):
            pieces.append(weight * _layer_norm(tensor, broadcast_mask, mask_float))
        return self.gamma * sum(pieces)


class RepresentationExtractor(nn.Module):
    def __init__(self, layer="last", pool="avg", layerwise_attention=None):
        super().__init__()
        self.layer = layer
        self.pool = pool
        self.layerwise_attention = layerwise_attention

    def forward(self, hidden_states, attention_mask):
        if self.layerwise_attention is not None:
            tok_emb = self.layerwise_attention(hidden_states, attention_mask)
        elif self.layer == "last":
            tok_emb = hidden_states[-1]
        else:
            tok_emb = hidden_states[int(self.layer)]

        if self.pool == "cls":
            return tok_emb[:, 0, :]
        if self.pool == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            return tok_emb.masked_fill(~mask, float("-inf")).max(dim=1).values
        mask = attention_mask.unsqueeze(-1)
        sent = (tok_emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return sent / denom
