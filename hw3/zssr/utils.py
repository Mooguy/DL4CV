import torch  # noqa
import torchvision  # noqa
import resize_right as rr
import interp_methods
from functools import partial
from pathlib import Path
from livelossplot import PlotLosses  

__all__ = ['rr_resize', 'rr_resize_by_shape', 'psnr', 'ROOT', 'DATA_ROOT']


#################################################
# PROVIDED: Constants
#################################################
ROOT = Path("/home/labs/antebilab/guyilan/Courses/DL4CV/hw3/zssr")
DATA_ROOT = ROOT / 'data'

#################################################
# PROVIDED: Resize Right Methods
#################################################
"""Resizes an image by the scale factor provided using the resize_right module.

  Args:
    image (torch.Tensor): Image to be resized. 
    Has shape `(batch_size, num_channels, height, width)`.
    scale_factor (float): Factor by which to resize. 

  Returns:
    resized_image (torch.Tensor): A resized image. 
    Has shape `(batch_size, num_channels, scale_factor * height, scale_factor * width)`.
  
  Example Usage:
    resized_image = rr_resize(image, scale_factors=2)
"""
rr_resize = partial(rr.resize, out_shape=None, support_sz=None, 
                    interp_method=interp_methods.cubic, antialiasing=True)

#################################################
# PROVIDED: PSNR evaluation method
#################################################
def psnr(im, ref, margin=2):
  """
   Args:
    im (torch.Tensor): Image to be evaluated. 
    Has shape `(num_channels, height, width)`.
    ref (torch.Tensor): reference image. 
    Has shape `(num_channels, height, width)`. 

  Returns:
    psnr (int): psnr value of the images.
  """
  # assume images are tensors float 0-1.
  # im, ref = (im*255).round(), (ref*255).round()
  rgb2gray = torch.Tensor([65.481, 128.553, 24.966]).to(im.device)[None, :, None, None]
  gray_im = torch.sum(im * rgb2gray, dim=1) + 16
  gray_ref = torch.sum(ref * rgb2gray, dim=1) + 16
  clipped_im = torch.clamp(gray_im, 0, 255).squeeze()
  clipped_ref = torch.clamp(gray_ref, 0, 255).squeeze()
  shaved_im = clipped_im[margin:-margin, margin:-margin]
  shaved_ref = clipped_ref[margin:-margin, margin:-margin]    
  return 20 * torch.log10(torch.tensor(255.)) -10.0 * ((shaved_im) - (shaved_ref)).pow(2.0).mean().log10()


##########################################################
# PROVIDED: Visualizer
##########################################################
class Visualizer:
  """Visualization using the liveplotloss library."""
  def __init__(self):
    self.liveloss = PlotLosses()

  def update(self, train_loss, train_psnr, val_psnr):
    """
    Args:
      
    Returns:
    """
    train_epoch_loss = train_loss  
    train_epoch_psnr = train_psnr  
    val_epoch_psnr = val_psnr
    
    if isinstance(train_epoch_loss, torch.Tensor):
      train_epoch_loss = train_epoch_loss.detach().cpu().numpy()
    if isinstance(train_epoch_psnr, torch.Tensor):
      train_epoch_psnr = train_epoch_psnr.detach().cpu().numpy()
    
    logs = {}
    logs[f'loss'] = train_epoch_loss
    logs[f'accuracy'] = train_epoch_psnr
    #logs[f'val_loss'] = val_epoch_loss
    logs[f'val_accuracy'] = val_epoch_psnr
    self.liveloss.update(logs)
    self.liveloss.send()
