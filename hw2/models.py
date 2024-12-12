import torch  # noqa

from nn import Module, Linear  # noqa
from functional import relu  # noqa

__all__ = ['SoftmaxClassifier', 'MLP']


#################################################
# SoftmaxClassifier
#################################################

class SoftmaxClassifier(Module):
  """A simple softmax classifier"""

  def __init__(self, in_dim, num_classes):
    super().__init__()
    # BEGIN SOLUTION
    self.fc = Linear(in_dim, num_classes)
    self._modules.append('fc')

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    x = x.view(x.size(0), -1)
    y = self.fc(x, ctx)
    return y
    # END SOLUTION


#################################################
# MLP
#################################################

class MLP(Module):
  """A multi-layer perceptron"""

  def __init__(self, in_dim, num_classes, hidden_dim=100):  # YOU CAN MODIFY THIS LINE AND ADD ARGUMENTS
    super().__init__()
    # BEGIN SOLUTION
    self.fc1 = Linear(in_dim, hidden_dim)
    self.fc2 = Linear(hidden_dim, num_classes)
    self._modules.extend(['fc1', 'fc2'])
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    x = x.view(x.size(0), -1)
    x = relu(self.fc1(x, ctx))
    y = self.fc2(x, ctx)
    return y
    # END SOLUTION

