import math  # noqa

import torch  # noqa
from torch.nn.modules.utils import _pair

from hw2_nn import Module
from functional import conv2d, max_pool2d  # noqa

from hw2_nn import *  # noqa
from hw2_nn import __all__ as __old_all__

__new_all__ = ['Conv2d', 'MaxPool2d']
__all__ = __old_all__ + __new_all__


class Conv2d(Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
               groups=1, bias=True):
    """Creates a Conv2d layer.

    In this method you should:
      * Create a weight parameter (call it `weight`).
      * Create a bias parameter (call it `bias`).
      * Add these parameter names to `self._parameters`.
      * Call `init_parameters()` to initialize the parameters.
      * Save the other arguments.

    Args:
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      kernel_size (Tuple[int, int] or int): Kernel Size.
      padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
        Defaults to 0.
      stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
        Defaults to 1.
      dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
        Defaults to 1.
      groups (int, Optional): Number of groups. Defaults to 1.
      bias (bool, optional): [description]. Defaults to True.
    """
    assert in_channels % groups == 0, \
      f'in_channels={in_channels} should be divisible by groups={groups}'
    assert out_channels % groups == 0,\
      f'out_channels={out_channels} should be divisible by groups={groups}'
    super().__init__()
    # BEGIN SOLUTION
    self.groups = groups
    self.in_channels = int(in_channels / self.groups)
    self.out_channels = out_channels
    self.kernal_y, self.kernal_x = _pair(kernel_size) if type(kernel_size) is int else kernel_size
    self.padding = padding
    self.stride = stride
    self.dilation = dilation

    self.weight = torch.zeros(self.out_channels, self.in_channels, self.kernal_y, self.kernal_x)
    self._parameters = ['weight']
    if bias:
      self.bias = torch.zeros(self.out_channels)
      self._parameters.append('bias')
    else:
      self.bias = None
    self.init_parameters()
    # END SOLUTION

  def init_parameters(self):
    """Initializes the layer's parameters."""
    # BEGIN SOLUTION
    n = self.in_channels * self.kernal_y * self.kernal_x
    stdv = 1. / math.sqrt(n)
    
    for param in self.parameters():
      param.uniform_(-stdv, stdv)
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the `conv2d` function of that input `x`.

    You should use the `weight` and `bias` parameters of that layer.

    Args:
      x (torch.Tensor): The input tensor. Has shape `(batch_size, in_channels, height, width)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor. Has shape `(batch_size, out_channels, out_height, out_width)`.
    """
    # BEGIN SOLUTION
    return conv2d(x, self.weight, b = self.bias, stride = self.stride,  padding = self.padding, dilation = self.dilation, 
                  groups = self.groups, ctx=ctx)
    # END SOLUTION


class MaxPool2d(Module):
  def __init__(self, kernel_size, padding=0, stride=1, dilation=1):
    """Creates a MaxPool2d layer.

    In this method you should:
      * Save the layer's arguments.

    Args:
      kernel_size (Tuple[int, int] or int): Kernel Size.
      padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
        Defaults to 0.
      stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
        Defaults to 1.
      dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
        Defaults to 1.
    """
    super().__init__()
    # BEGIN SOLUTION
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    self.dilation = dilation

    self._parameters = []
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the `max_pool2d` function of that input `x`.

    Args:
      x (torch.Tensor): The input tensor. Has shape `(batch_size, in_channels, height, width)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor. Has shape `(batch_size, in_channels, out_height, out_width)`.
    """
    # BEGIN SOLUTION
    return max_pool2d(x, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                      dilation = self.dilation, ctx=ctx)
    # END SOLUTION

