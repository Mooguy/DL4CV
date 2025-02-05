import torch  # noqa
from torch.nn.functional import one_hot  # noqa

__all__ = ['linear', 'relu', 'softmax', 'cross_entropy', 'cross_entropy_loss']


#################################################
# EXAMPLE: mean
#################################################

def mean(x, ctx=None):
  """A differentiable Mean function.

  Backward call:
    backward_fn: mean_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output scalar tensor, the mean of `x`.
  """
  y = x.mean()
  # the backward function with its arguments is appended to `ctx`
  if ctx is not None:
    ctx.append([mean_backward, [y, x]])
  return y


def mean_backward(y, x):
  """Backward computation of `mean`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output scalar tensor.
    x (torch.Tensor): The input tensor.
  """
  # the gradient of `x` is added to `x.grad`
  x.grad += torch.ones_like(x) * (y.grad / x.numel())


#################################################
# linear
#################################################

def linear(x, w, b, ctx=None):
  """A differentiable Linear function. Computes: y = w * x + b

  Backward call:
    backward_fn: linear_backward
    args: y, x, w, b

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(out_dim, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(out_dim,)`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor, has shape `(batch_size, out_dim)`.
  """
  # VECTORIZATION HINT: torch.mm

  # BEGIN SOLUTION
  y = torch.mm(x, w.T) + b

  if ctx is not None:
    ctx.append([linear_backward, [y, x, w, b]])

  return y
  # END SOLUTION

def linear_backward(y, x, w, b):
    """Backward computation of `linear`.

    Propagates the gradients of `y` (in `y.grad`) to `x`, `w` and `b`,
    and accumulates them in `x.grad`, `w.grad` and `b.grad` respectively.

    Args:
      y (torch.Tensor): The output tensor, has shape `(batch_size, out_dim)`.
      x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
      w (torch.Tensor): The weight tensor, has shape `(out_dim, in_dim)`.
      b (torch.Tensor): The bias tensor, has shape `(out_dim,)`.
    """

    # BEGIN SOLUTION
    x.grad += torch.mm(y.grad, w)

    w.grad += torch.mm(y.grad.T, x)

    b.grad += torch.sum(y.grad, dim=0)
    # END SOLUTION

#################################################
# relu
#################################################

def relu(x, ctx=None):
  """A differentiable ReLU function.

  Note: `y` should be a different tensor than `x`. `x` should not be changed.
        Read about Tensor.clone().

  Note: Don't modify the input in-place.

  Backward call:
    backward_fn: relu_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. Has non-negative values.
  """
  # BEGIN SOLUTION
  y = x.clamp(min=0)

  if ctx is not None:
    ctx.append([relu_backward, [y, x]])

  return y
  # END SOLUTION


def relu_backward(y, x):
  """Backward computation of `relu`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor. Has non-negative values.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  x.grad += (x > 0).float() * y.grad
  # END SOLUTION


#################################################
# softmax
#################################################

def softmax(x, ctx=None):
  """A differentiable Softmax function.

  Note: make sure to add `x` from the input to the context,
        and not some intermediate tensor.

  Backward call:
    backward_fn: softmax_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor, has shape `(batch_size, num_classes)`.
      Each row in `y` is a probability distribution over the classes.
  """
  # BEGIN SOLUTION
  max_values, _ = torch.max(x, axis=1, keepdims=True)

  exp_shifted = torch.exp(x - max_values)

  y = exp_shifted / torch.sum(exp_shifted, axis=1, keepdims=True)

  if ctx is not None:
    ctx.append([softmax_backward, [y, x]])

  return y
  # END SOLUTION


def softmax_backward(y, x):
  """Backward computation of `softmax`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor, has shape `(batch_size, num_classes)`.
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.
  """
  # VECTORIZATION HINT: one_hot, torch.gather, torch.einsum

  # BEGIN SOLUTION

  batch_size, num_classes = y.shape
  y_diag = torch.eye(num_classes, device=y.device).unsqueeze(0) * y.unsqueeze(2)  

  y_outer = torch.einsum('ij,ik->ijk', y, y)

  jacobian = y_diag - y_outer 

  x_grad = torch.einsum('bij,bj->bi', jacobian, y.grad) 

  x.grad += x_grad

  # END SOLUTION


#################################################
# cross_entropy
#################################################

def cross_entropy(pred, target, ctx=None):
  """A differentiable Cross-Entropy function for hard-labels.

  Backward call:
    backward_fn: cross_entropy
    args: loss, pred, target

  Args:
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Each row is a probability distribution over the classes.
    target (torch.Tensor): The targets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    loss (torch.Tensor): The per-example cross-entropy tensor, has shape `(batch_size,).
      Each value is the cross-entropy loss of that example in the batch.
  """
  # VECTORIZATION HINT: one_hot, torch.gather
  eps = torch.finfo(pred.dtype).tiny
  # BEGIN SOLUTION

  pred_log = torch.log(pred + eps) 
  
  loss = -pred_log.gather(1, target.unsqueeze(1)).squeeze(1)

  if ctx is not None:
    ctx.append([cross_entropy_backward, [loss, pred, target]])
  
  return loss
  # END SOLUTION

#################################################
# cross_entropy_
#################################################

def cross_entropy_backward(loss, pred, target):
  """Backward computation of `cross_entropy`.

  Propagates the gradients of `loss` (in `loss.grad`) to `pred`,
  and accumulates them in `pred.grad`.

  Note: `target` is an integer tensor and has no gradients.

  Args:
    loss (torch.Tensor): The per-example cross-entropy tensor, has shape `(batch_size,).
      Each value is the cross-entropy loss of that example in the batch.
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Each row is a probability distribution over the classes.
    target (torch.Tensor): The tragets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
  """
  # VECTORIZATION HINT: one_hot, torch.gather, torch.scatter_add
  eps = torch.finfo(pred.dtype).tiny
  # BEGIN SOLUTION
  batch_size = pred.size(0)
    
  if pred.grad is None:
      pred.grad = torch.zeros_like(pred)
  
  loss_grad = loss.grad if loss.grad is not None else torch.ones_like(loss)
  
  target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).to(pred.dtype)
  grad = -target_one_hot / (pred + eps) 
  grad *= loss_grad.unsqueeze(1)       
  grad /= 1                
  
  pred.grad += grad
  # END SOLUTION


#################################################
# PROVIDED: cross_entropy_loss
#################################################

def cross_entropy_loss(pred, target, ctx=None):
  """A differentiable Cross-Entropy loss for hard-labels.

  This differentiable function is similar to PyTorch's cross-entropy function.

  Note: Unlike `cross_entropy` this function expects `pred` to be BEFORE softmax.

  Note: You should not implement the backward of that function explicitly, as you use only
        differentiable functions for that. That part of the "magic" in autograd --
        you can simply compose differentiable functions, and it works!

  Args:
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Unlike `cross_entropy`, this prediction IS NOT a probability distribution over
      the classes. It expects to see predictions BEFORE sofmax.
    target (torch.Tensor): The targets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    loss (torch.Tensor): The scalar loss tensor. The mean loss over the batch.
  """
  pred = softmax(pred, ctx=ctx)
  batched_loss = cross_entropy(pred, target, ctx=ctx)
  loss = mean(batched_loss, ctx=ctx)
  return loss

