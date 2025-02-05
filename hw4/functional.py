from torch.nn.functional import softmax  # noqa
from torch import bmm, cat, transpose, reshape, matmul  # noqa
import torch

__all__ = ["multi_head_attention"]


#################################################
# Multi Head Attention
#################################################


def multi_head_attention(q, k, v):
    """A differentiable multi head attention function.

    Args:
      q (torch.Tensor): The query embedding.
        Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
      k (torch.Tensor): The key embedding.
        Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
      v (torch.Tensor): The value embedding.
        Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.

    Returns:
      y (torch.Tensor): The multi head attention output.
        Has shape `(batch_size, sequence_size, num_heads * head_emb_dim)`.
    """
    # BEGIN SOLUTION
    batch_size, num_heads, sequence_size, head_emb_dim = q.shape

    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(head_emb_dim, dtype=torch.float32)
    )

    attn_weights = softmax(scores, dim=-1)

    attn_output = torch.matmul(attn_weights, v)

    y = (
        attn_output.transpose(1, 2)
        .contiguous()
        .view(batch_size, sequence_size, num_heads * head_emb_dim)
    )

    return y
    # END SOLUTION
