import torch  # noqa

from torch.nn import Module, Linear  # noqa
from torch.nn.functional import relu  # noqa

__all__ = ['SoftmaxClassifier', 'MLP']


#################################################
# SoftmaxClassifier
#################################################

class SoftmaxClassifier(Module):
  """A simple softmax classifier"""

  def __init__(self, in_dim, num_classes):
    super().__init__()  # This line is important in torch Modules.
                        # It replaces the manual registration of parameters
                        # and sub-modules (to some extent).
    # BEGIN SOLUTION
    self.linear = Linear(in_dim, num_classes)
    # END SOLUTION

  def forward(self, x):
    """Computes the forward function of the network.

    Note: `F.cross_entropy` expects predictions BEFORE applying `F.softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    x = x.view(x.size(0), -1)
    y = self.linear(x)
    return y
    # END SOLUTION


#################################################
# MLP
#################################################

class MLP(Module):
  """A multi-layer perceptron"""

  def __init__(self, in_dim, num_classes):  # YOU CAN MODIFY THIS LINE AND ADD ARGUMENTS
    super().__init__()  # This line is important in torch Modules.
                        # It replaces the manual registration of parameters
                        # and sub-modules (to some extent).
    # BEGIN SOLUTION
    self.fc1 = Linear(in_dim, 64)
    self.fc2 = Linear(64, 32)
    self.fc3 = Linear(32, num_classes)

    self.relu = relu
    # END SOLUTION

  def forward(self, x):
    """Computes the forward function of the network.

    Note: `F.cross_entropy` expects predictions BEFORE applying `F.softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    y = self.fc3(x)

    return y
    # END SOLUTION
