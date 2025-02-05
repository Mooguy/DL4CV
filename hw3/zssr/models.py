from torch import nn
import utils
from functools import partial

##########################################################
# Basic Model
##########################################################
class ZSSRNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """
    # BEGIN SOLUTION
    super(ZSSRNet, self).__init__()

    self.s = scale_factor
      
    layers = []
    
    layers.append(nn.Conv2d(3, 64, kernel_size, padding=kernel_size//2))
    layers.append(nn.ReLU(inplace=True))
    
    for _ in range(6):
        layers.append(nn.Conv2d(64, 64, kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(64, 3, kernel_size, padding=kernel_size//2))
    
    self.net = nn.Sequential(*layers)
    #END SOLUTION

  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """    
    # BEGIN SOLUTION
    x_up = utils.rr_resize(x, self.s)
    output = self.net(x_up)
    return output
    # END SOLUTION


##########################################################
# Advanced Model
##########################################################
class ZSSRResNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """
    # BEGIN SOLUTION
    raise NotImplementedError
    # END SOLUTION

  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Finally, add the CNN's output in a residual manner to the original resized
    image.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """   
    # BEGIN SOLUTION
    raise NotImplementedError
    # END SOLUTION


##########################################################
# Original Model
##########################################################
class ZSSROriginalNet(nn.Module):
  """A super resolution model. """

  def __init__(self, **kwargs):
    # BEGIN SOLUTION
    raise NotImplementedError     
    # END SOLUTION

  def forward(self, x):
    # BEGIN SOLUTION
    raise NotImplementedError     
    # BEGIN SOLUTION
