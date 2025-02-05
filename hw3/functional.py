import math  # noqa

import torch  # noqa
from torch.nn.functional import one_hot  # noqa
from torch.nn.modules.utils import _pair, _ntuple
from torch.nn.functional import unfold, fold

from hw2_functional import *  # noqa
from hw2_functional import __all__ as __old_all__

__new_all__ = ['view', 'add', 'conv2d', 'max_pool2d']
__all__ = __old_all__ + __new_all__ # 


#################################################
# conv2d
#################################################
def conv2d(x, w, b=None, padding=0, stride=1, dilation=1, groups=1, ctx=None):
  """A differentiable convolution of 2d tensors.

  Note: Read this following documentation regarding the output's shape.
  https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

  Backward call:
    backward_fn: conv2d_backward
    args: y, x, w, b, padding, stride, dilation

  Args:
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    w (torch.Tensor): The weight tensor.
      Has shape `(out_channels, in_channels, kernel_height, kernel_width)`.
    b (torch.Tensor): The bias tensor. Has shape `(out_channels,)`.
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    groups (int, Optional): Number of groups. Defaults to 1.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, out_channels, out_put, out_width)`.
  """
  assert w.size(0) % groups == 0, \
    f'expected w.size(0)={w.size(0)} to be divisible by groups={groups}'
  assert x.size(1) % groups == 0, \
    f'expected x.size(1)={x.size(1)} to be divisible by groups={groups}'
  assert x.size(1) // groups == w.size(1), \
    f'expected w.size(1)={w.size(1)} to be x.size(1)//groups={x.size(1)}//{groups}'

  # BEGIN SOLUTION
  if b is None:
    b = torch.zeros(w.size(0), dtype=x.dtype, device=x.device)
  
  #Extracting the dimensions of the input and the filter
  input_dims = x.shape
  filter_dims = w.shape
  n_samples = input_dims[0]
  output_channels = filter_dims[0]
  input_channels = filter_dims[1] * groups
  
  #Extracting the padding, stride and dilation
  pad_y, pad_x = _pair(padding) if isinstance(padding, int) else padding
  step_y, step_x = _pair(stride) if isinstance(stride, int) else stride
  dil_y, _dil_x = _pair(dilation) if isinstance(dilation, int) else dilation
  
  #Calculating the output height and width
  kernel_y, kernel_x = filter_dims[2:]
  output_y = int(1 + ((input_dims[2] + 2*pad_y - dil_y * (kernel_y - 1) - 1) / step_y))
  output_x = int(1 + ((input_dims[3] + 2*pad_x - _dil_x * (kernel_x - 1) - 1) / step_x))

  #Splitting the input and the filter into groups
  input_matrix = unfold(x, (kernel_y, kernel_x),
                       dilation=dilation,
                       padding=padding,
                       stride=stride)
  
  reshaped_input = input_matrix.reshape(n_samples, input_channels, 
                                      kernel_y*kernel_x, output_y*output_x)
  grouped_input = torch.stack(reshaped_input.chunk(chunks=groups, dim=1), dim=1)
  grouped_input = grouped_input.reshape(n_samples, groups, 
                                      int(input_channels/groups)*kernel_y*kernel_x, 
                                      output_y*output_x)

  reshaped_weights = w.reshape(output_channels, 
                              int(input_channels/groups)*kernel_y*kernel_x)
  grouped_weights = torch.stack(reshaped_weights.chunk(chunks=groups, dim=0))
  grouped_weights = grouped_weights.reshape(1, groups, 
                                          int(output_channels/groups), 
                                          int(input_channels/groups)*kernel_y*kernel_x)
  
  #Performing the convolution!!!
  result = torch.matmul(grouped_weights, grouped_input)
  output = result.reshape(n_samples, output_channels, output_y, output_x)
  output += b.reshape(1, -1, 1, 1)

  if ctx is not None:
    ctx += [(conv2d_backward, [output, x, w, b, padding, stride, dilation, groups])]
  
  return output
# END SOLUTION

def conv2d_backward(y, x, w, b, padding, stride, dilation, groups):
  """Backward computation of `conv2d`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, `w` and `b` (if `b` is not None),
  and accumulates them in `x.grad`, `w.grad` and `b.grad`, respectively.

  Args:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, out_channels, out_put, out_width)`.
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    w (torch.Tensor): The weight tensor.
      Has shape `(out_channels, in_channels, kernel_height, kernel_width)`.
    b (torch.Tensor): The bias tensor. Has shape `(out_channels,)`.
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    groups (int, Optional): Number of groups. Defaults to 1.
  """
# BEGIN SOLUTION
  input_dims = x.shape
  filter_dims = w.shape
  n_samples = input_dims[0]
  output_channels = filter_dims[0]
  input_channels = filter_dims[1] * groups
  
  pad_y, pad_x = _pair(padding) if isinstance(padding, int) else padding
  step_y, step_x = _pair(stride) if isinstance(stride, int) else stride
  dil_y, _dil_x = _pair(dilation) if isinstance(dilation, int) else dilation
  

  kernel_y, kernel_x = filter_dims[2:]
  output_y = int(1 + ((input_dims[2] + 2 *pad_y - dil_y * (kernel_y - 1) - 1) / step_y))
  output_x = int(1 + ((input_dims[3] + 2 *pad_x - _dil_x * (kernel_x - 1) - 1) / step_x))


  unfolded = unfold(x, (kernel_y, kernel_x),
                   dilation=dilation,
                   padding=padding,
                   stride=stride)
  
  matrix_form = unfolded.reshape(n_samples, input_channels, 
                                kernel_y*kernel_x, output_y*output_x)
  transposed = matrix_form.permute([0, 3, 1, 2])
  split_patches = torch.stack(transposed.chunk(chunks=groups, dim=2), dim=0)
  transformed_patches = split_patches.reshape(groups, 
                                            n_samples*output_y*output_x,
                                            int(input_channels/groups)*kernel_y*kernel_x)

  b.grad += y.grad.sum([0, 2, 3])

  #Computing the gradients of the weights
  grad_reshaped = y.grad.permute([1, 0, 2, 3])
  grad_matrix = grad_reshaped.reshape(output_channels, n_samples*output_y*output_x)
  split_grads = torch.stack(grad_matrix.chunk(chunks=groups, dim=0), dim=0)
  weight_gradients = split_grads.matmul(transformed_patches)
  w.grad += weight_gradients.reshape(output_channels, int(input_channels/groups), 
                                   kernel_y, kernel_x)

  #Computing the gradients of the input
  transposed_grads = split_grads.permute([0, 2, 1])
  reshaped_weights = w.reshape(output_channels, -1)
  split_weights = torch.stack(reshaped_weights.chunk(chunks=groups, dim=0), dim=0)
  patch_gradients = transposed_grads.matmul(split_weights)

  #Reshaping the gradients to the original input shape
  reshaped_gradients = patch_gradients.reshape(groups, 
                                             n_samples*output_y*output_x,
                                             int(input_channels/groups),
                                             kernel_y*kernel_x)
  permuted_gradients = reshaped_gradients.permute([1, 0, 2, 3])
  final_reshape = permuted_gradients.reshape(n_samples, 
                                           output_y*output_x,
                                           input_channels*kernel_y*kernel_x)
  input_gradients = final_reshape.permute([0, 2, 1])

  #Accumulating the gradients in the input
  x.grad += fold(input_gradients,
                output_size=(input_dims[2], input_dims[3]),
                kernel_size=(kernel_y, kernel_x),
                dilation=dilation,
                padding=padding,
                stride=stride)
# END SOLUTION


#################################################
# max_pool2d
#################################################

def max_pool2d(x, kernel_size, padding=0, stride=1, dilation=1, ctx=None):
  """A differentiable convolution of 2d tensors.

  Note: Read this following documentation regarding the output's shape.
  https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

  Backward call:
    backward_fn: max_pool2d_backward
    args: y, x, padding, stride, dilation

  Args:
    x (torch.Tensor): The input tensor. Has shape `(batch_size, in_channels, height, width)`.
    kernel_size (Tuple[int, int] or int): The kernel size in each dimension (height, width).
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, in_channels, out_put, out_width)`.
  """
  # BEGIN SOLUTION
  input_dims = x.shape
  kernel_y, kernel_x = _pair(kernel_size) if isinstance(kernel_size, int) else kernel_size
  pad_y, pad_x = _pair(padding) if isinstance(padding, int) else padding
  step_y, step_x = _pair(stride) if isinstance(stride, int) else stride
  dil_y, dil_x = _pair(dilation) if isinstance(dilation, int) else dilation

  output_y = int(1 + ((input_dims[2] + 2 * pad_y - dil_y * (kernel_y - 1) - 1) / (step_y)))
  output_x = int(1 + ((input_dims[3] + 2 * pad_x - dil_x * (kernel_x - 1) - 1) / (step_x)))

  input_matrix = unfold(x, (kernel_y, kernel_x),
                    dilation=dilation,
                    padding=padding,
                    stride=stride)
  input_matrix = input_matrix.reshape(input_dims[0], input_dims[1], kernel_y * kernel_x, output_y, output_x)

  y, index = input_matrix.max(dim=2)

  if ctx is not None:
      ctx += [(max_pool2d_backward, [y, x, index, kernel_size, padding, stride, dilation])]

  return y
  # END SOLUTION

def max_pool2d_backward(y, x, index, kernel_size, padding, stride, dilation):
  """Backward computation of `max_pool2d`.

  Propagates the gradients of `y` (in `y.grad`) to `x` and accumulates it in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, in_channels, out_put, out_width)`.
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    index (torch.Tensor): Auxilary tensor with indices of the maximum elements. You are
      not restricted to a specific format.
    kernel_size (Tuple[int, int] or int): The kernel size in each dimension (height, width).
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
  """
  # BEGIN SOLUTION
  input_dims = x.shape
  batch_size, in_channels, output_y, output_x = y.shape
  kernel_y, kernel_x = _pair(kernel_size) if isinstance(kernel_size, int) else kernel_size

  one_hot_entries = one_hot(index, num_classes=kernel_y * kernel_x)
  one_hot_entries = one_hot_entries.permute([0, 1, 4, 2, 3])

  permuted_x_grad = one_hot_entries * y.grad.unsqueeze(2)
  permuted_x_grad = permuted_x_grad.reshape(batch_size, in_channels * kernel_y * kernel_x, output_y * output_x)

  x.grad += fold(permuted_x_grad,
                output_size=(input_dims[2], input_dims[3]),
                kernel_size=(kernel_y, kernel_x),
                dilation=dilation,
                padding=padding,
                stride=stride)
  # END SOLUTION
  
#################################################
# view
#################################################

def view(x, size, ctx=None):
  """A differentiable view function.
t
  Backward call:
    backward_fn: view_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    size (torch.Size): The new size (shape).
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. Has shape `size`.
  """
  # BEGIN SOLUTION
  y = x.view(size)
  if ctx is not None:
    ctx += [(view_backward, [y, x])]
  return y
  # END SOLUTION


def view_backward(y, x):
  """Backward computation of `view`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  x.grad += y.grad.view_as(x)
  # END SOLUTION


#################################################
# add
#################################################

def add(a, b, ctx=None):
  """A differentiable addition of two tensors.

  Backward call:
    backward_fn: add_backward
    args: y, a, b

  Args:
    a (torch.Tensor): The first input tensor.
    b (torch.Tensor): The second input tensor. Should have the same shape as `a`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. The sum of `a + b`.
  """
  assert a.size() == b.size(), 'tensors should have the same size'
  # BEGIN SOLUTION
  y = a + b
  if ctx is not None:
    ctx += [(add_backward, [y, a, b])]
  return y
  # END SOLUTION
  

def add_backward(y, a, b):
  """Backward computation of `add`.

  Propagates the gradients of `y` (in `y.grad`) to `a` and `b`, and accumulates them in `a.grad`,
  `b.grad`, respectively.

  Args:
    y (torch.Tensor): The output tensor.
    a (torch.Tensor): The first input tensor.
    b (torch.Tensor): The second input tensor.
  """
  # BEGIN SOLUTION
  a.grad += y.grad
  b.grad += y.grad
  # END SOLUTION
