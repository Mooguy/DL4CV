import torch  # noqa

from nn import Module, Linear, Conv2d, MaxPool2d  # noqa
from functional import relu, view, add # noqa

__all__ = ['ConvNet']


#################################################
# ConvNet
#################################################

class ConvNet(Module):
  """A deep convolutional neural network"""

  def __init__(self, in_channels, num_classes):
    super().__init__()
    self.conv1 = Conv2d(in_channels, 32, kernel_size=3, padding=1)
    self.conv2 = Conv2d(32, 32, kernel_size=3, padding=1)
    self.max1 = MaxPool2d(kernel_size=2, stride=2)
    
    self.conv3 = Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv4 = Conv2d(64, 64, kernel_size=3, padding=1)
    self.max2 = MaxPool2d(kernel_size=2, stride=2)
    
    self.conv5 = Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv6 = Conv2d(128, 128, kernel_size=3, padding=1)
    self.max3 = MaxPool2d(kernel_size=2, stride=2)

    # Final fully connected layer
    self.linear = Linear(128 * 4 * 4, num_classes)

    self._modules = ['conv1', 'conv2', 'max1', 'conv3', 'conv4', 'max2', 'conv5', 'conv6', 'max3', 'linear']
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_channels, height, width)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): Thexput tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    x = relu(self.conv1(x, ctx=ctx), ctx=ctx)
    x = relu(self.conv2(x, ctx=ctx), ctx=ctx)
    x = self.max1(x, ctx=ctx)

    x = relu(self.conv3(x, ctx=ctx), ctx=ctx)
    x = relu(self.conv4(x, ctx=ctx), ctx=ctx)
    x = self.max2(x, ctx=ctx)

    x = relu(self.conv5(x, ctx=ctx), ctx=ctx)
    x = relu(self.conv6(x, ctx=ctx), ctx=ctx)
    x = self.max3(x, ctx=ctx)

    x = view(x, (x.size(0), -1), ctx=ctx)  # Flatten for the fully connected layer
    x = self.linear(x, ctx=ctx)
    return x
    # END SOLUTION
