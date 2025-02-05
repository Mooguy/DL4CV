import math  # noqa
from functional import multi_head_attention  # noqa
from torch.nn import Module, Linear, Conv2d, LayerNorm, BatchNorm1d, Parameter  # noqa
from torch import unbind, reshape, permute, flatten, zeros, empty, chunk  # noqa
from torch.nn.init import trunc_normal_  # noqa
from vit_helpers import Mlp  # noqa

__all__ = [
    "MHSA",
    "TransformerBlock",
    "PatchEmbedding",
    "PositionalEmbedding",
    "CLSToken",
]


#################################################
# Multi Head Self Attention Layer
#################################################


class MHSA(Module):
    def __init__(self, dim, num_heads):
        """Creates a Multi Head Self Attention layer.

        Args:
          dim (int): The input and output dimension (in this implementation Dy=Dq=Dk=Dv=Dx)
          num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # BEGIN SOLUTION
        self.lin_qkv = Linear(dim, 3 * dim)  # use this variable name
        self.lin = Linear(dim, dim)  # use this variable name

        trunc_normal_(self.lin_qkv.weight)
        self.lin_qkv.bias.data.fill_(0)
        trunc_normal_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        # END SOLUTION

    def forward(self, x):
        """Computes the `MHSA` of the input `x`.

        Args:
          x (torch.Tensor): The input tensor.
            Has shape `(batch_size, sequence_size, dim)`.

        Returns:
          y (torch.Tensor): The output tensor.
            Has shape `(batch_size, sequence_size, dim)`.
        """
        # BEGIN SOLUTION
        batch_size, seq_len, _ = x.shape

        # Compute q, k, v
        qkv = self.lin_qkv(x)  # Shape: (batch_size, sequence_size, 3 * dim)
        q, k, v = chunk(
            qkv, 3, dim=-1
        )  # Each has shape (batch_size, sequence_size, dim)

        # Reshape into (batch_size, sequence_size, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, sequence_size, head_dim) for matmul
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = multi_head_attention(q, k, v)

        # Final projection
        y = self.lin(attn_output)
        return y


#################################################
# Transformer Block
#################################################
class TransformerBlock(Module):
    def __init__(self, dim, num_heads):
        """Creates a transformer block

        Args:
          dim (int): The input dimension
          num_heads (int): Number of attention heads.

        ***Note*** Do not waste time implementing an MLP. An implementation is
        already provided to you (see vit_helpers->Mlp).
        """
        super().__init__()
        # BEGIN SOLUTION
        self.norm1 = LayerNorm(dim)  # use this variable name
        self.mhsa = MHSA(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, dim)
        # END SOLUTION

    def forward(self, x):
        """Apply a transfomer block on an input `x`.

        Args:
          x (torch.Tensor): The input tensor.
            Has shape `(batch_size, sequence_size, dim)`.

        Returns:
          y (torch.Tensor): The output tensor.
            Has shape `(batch_size, sequence_size, dim)`.
        """
        # BEGIN SOLUTION
        x1 = self.mhsa(self.norm1(x))
        x = x + x1
        x1 = self.mlp(self.norm2(x))
        return x + x1
        # END SOLUTION


#################################################
# Patch Embedding
#################################################


class PatchEmbedding(Module):
    """Divide an image into patches and project them to a given dimension."""

    def __init__(self, patch_dim, in_chans, dim):
        """Creates a PatchEmbedding layer.

        Args:
          patch_dim (int): Patch dim, we use only squared patches and squared
            images so the total patch size is (patch_dim, patch_dim).
          in_chans (int): Number of channels in the input image
          dim (int): The projection output dimension.
        """
        super().__init__()
        # BEGIN SOLUTION
        self.patch_embed = Conv2d(
            in_chans, dim, kernel_size=patch_dim, stride=patch_dim
        )
        # END SOLUTION

    def forward(self, x):
        """Divide an image into patches and project them.

        Args:
          x (torch.Tensor): The input image.
            Has shape `(batch_size, in_chans, img_dim, img_dim)`, we use only squared images.

        Returns:
          y (torch.Tensor): The output tensor.
            Has shape `(batch_size, sequence_size, dim)`.
        """
        # BEGIN SOLUTION
        y = self.patch_embed(x).flatten(2).transpose(1, 2)
        return y
        # END SOLUTION

    @staticmethod
    def get_sequence_size(img_dim, patch_dim):
        """Calculate the number of patches

        Args:
          img_dim (int): Image dim, we use only squared images so the total
            image size is (in_chans, img_dim, img_dim).
          patch_dim (int): Patch dim, we use only squared patches so the total
            patch size is (patch_dim, patch_dim).
        """
        # BEGIN SOLUTION
        return (img_dim // patch_dim) * 2
        # END SOLUTION


#################################################
# Positional Embedding
#################################################
class PositionalEmbedding(Module):
    def __init__(self, sequence_size, dim, init_std):
        """Creates a PositionalEmbedding.

        Args:
          sequence_size (int): The sequence size.
          dim (int): The positional embedding dimension.
          init_std (int): The standard deviation of the truncated normal
            distribution used for initialization.

        **Important note:**
        You may not use PyTorch's nn.Embedding layer.
        Instead, create your own tensor to be the learned parameters,
        and don't forget to wrap it with PyTorch's nn.Parameter
        """
        super().__init__()
        # BEGIN SOLUTION
        emp_seq = empty(1, sequence_size, dim)
        self.pos_emb = Parameter(trunc_normal_(emp_seq, std=init_std))
        # END SOLUTION

    def forward(self):
        """Return the positional embedding.

        Returns:
          y (torch.Tensor): The embedding tensor.
            Has shape `(1, sequence_size, dim)`.
        """
        # BEGIN SOLUTION
        return self.pos_emb
        # END SOLUTION


#################################################
# CLS Token
#################################################
class CLSToken(Module):
    def __init__(self, dim, init_std):
        """Creates a CLSToken.

        Args:
          dim (int): The token dimension.
          init_std (int): The standard deviation of the truncated normal
            distribution used for initialization.

        **Important note:**
        You may not use PyTorch's nn.Embedding layer.
        Instead, create your own tensor to be the learned parameters,
        and don't forget to wrap it with PyTorch's nn.Parameter
        """
        super().__init__()
        # BEGIN SOLUTION
        emp_tokens = zeros(1, 1, dim)
        self.cls = Parameter(trunc_normal_(emp_tokens, std=init_std))
        # END SOLUTION

    def forward(self):
        """Returns the Class Token.

        Returns:
          y (torch.Tensor): The token tensor.
            Has shape `(1, 1, dim)`.
        """
        # BEGIN SOLUTION
        return self.cls
        # END SOLUTION
