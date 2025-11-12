import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from timm.models.vision_transformer import Block

class ReverseLayerF(Function):
    """
    Gradient reversal layer for domain adaptation.
    """
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLP(torch.nn.Module):
    """
    Multi-layer perceptron (MLP) module.
    """
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class cnn_extractor(nn.Module):
    """
    CNN feature extractor for sequence data.
    """
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

class MaskAttention(torch.nn.Module):
    """
    Compute attention layer with optional mask.
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores

class Attention(torch.nn.Module):
    """
    Scaled dot-product attention.
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    """
    Multi-headed attention mechanism.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn

class SelfAttentionFeatureExtract(torch.nn.Module):
    """
    Self-attention feature extraction module.
    """
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
    def forward(self, inputs, query, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        feature, attn = self.attention(query=query,
                                 value=inputs,
                                 key=inputs,
                                 mask=mask
                                 )
        return feature, attn

def masked_softmax(scores, mask):
    """
    Softmax with mask support.
    """
    scores = scores.masked_fill(mask == 0, -np.inf)
    return F.softmax(scores.float(), dim=-1).type_as(scores)
 
class ParallelCoAttentionNetwork(nn.Module):
    """
    Parallel co-attention network for multi-modal fusion.
    """
    def __init__(self, hidden_dim, co_attention_dim, mask_in=False):
        super(ParallelCoAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.mask_in = mask_in
        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
 
    def forward(self, V, Q, V_mask=None, Q_mask=None):
        # Compute affinity matrix
        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # Compute hidden representations
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
        # Attention weights
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)

        if self.mask_in:
            # Apply mask to attention weights
            masked_a_v = masked_softmax(
                a_v.squeeze(1), V_mask
            ).unsqueeze(1)

            masked_a_q = masked_softmax(
                a_q.squeeze(1), Q_mask
            ).unsqueeze(1)

            v = torch.squeeze(torch.matmul(masked_a_v, V.permute(0, 2, 1)))
            q = torch.squeeze(torch.matmul(masked_a_q, Q))
    
            return masked_a_v, masked_a_q, v, q
        else:
            v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
            q = torch.squeeze(torch.matmul(a_q, Q))
    
            return a_v, a_q, v, q