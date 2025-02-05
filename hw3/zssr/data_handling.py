import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image
import utils


##########################################################
# Dataset
##########################################################

class BasicZSSRDataset(Dataset):
  """ZSSR dataset. Creates a pair of LR and SR images. 
  The LR image is used for training while the SR image is used solely for
  evaluation."""

  def __init__(self, image_path, scale_factor=2, transform=transforms.ToTensor()):
    """
    Args:            
      image_path (string): Path to image from which to create dataset
      scale_factor (int): Ratio between SR and LR images.
      transform (callable): Transform to be applied on a sample. 
      Default transform turns the image into a tensor.
    """
    self.image_path = image_path
    self.scale_factor = scale_factor
    self.transform = transform
    # END SOLUTION

  def __len__(self):
      """
      Returns:
        the length of the dataset. 
        Our dataset contains a single pair so the length is 1.
      """
      # BEGIN SOLUTION
      return 1
      # END SOLUTION


  def __getitem__(self, idx):
    """
    Args:
      idx (int) - Index of element to fetch. In our case only 1.
    Returns:
      sample (dict) - a dictionary containing two elements:
      Under the key 'SR' a torch.Tensor representing the original image.
      Has shape `(num_channels, height, width)`.
      Under the key 'LR' a torch.Tensor representing the original image.
      Has shape `(num_channels, height / scale_factor, width / scale_factor)`.
      In our case, returns the only element in the dataset.
    """
    # BEGIN SOLUTION
    assert idx == 0
    image = self.transform(Image.open(self.image_path))
    DB = [
      { 'SR': image, 
        'LR': utils.rr_resize(image.unsqueeze(0), 1 / self.scale_factor).squeeze()}
    ]
    return DB[idx]
    # END SOLUTION

class OriginalZSSRDataset(Dataset):
  """Your original ZSSR Dataset. Must include a ground truth SR image and a
  LR image for training"""

  def __init__(self, image_path, scale_factor=2, transform=transforms.ToTensor()):
      """
      Args:            
        image_path (string): Path to image from which to create dataset
        scale_factor (int): Ratio between SR and LR images.
        transform (callable): Transform to be applied on a sample. 
        Default transform turns the image into a tensor.
      """
      # BEGIN SOLUTION
      raise NotImplementedError     
      # END SOLUTION

  def __len__(self):
      """
      Returns:
        the length of the dataset. 
        Our dataset contains a single pair so the length is 1.
      """
      # BEGIN SOLUTION
      raise NotImplementedError     
      # END SOLUTION

  def __getitem__(self, idx):
      """
      Args:
        idx (int) - Index of element to fetch.
      Returns:
        sample (dict) - a dictionary containing two elements:
        Under the key 'SR' a torch.Tensor representing the original image.
        Has shape `(num_channels, height, width)`.
        Under the key 'LR' a torch.Tensor representing the original image.
        Has shape `(num_channels, height / scale_factor, width / scale_factor)`.
      """
      # BEGIN SOLUTION
      raise NotImplementedError
      # END SOLUTION

##########################################################
# Transforms 
##########################################################

class EightCrops:
  """Generate all the possible crops using combinations of
  [90, 180, 270 degrees rotations, horizontal flips and vertical flips]. 
  In total there are 8 options."""

  def __init__(self):
    pass

  def __call__(self, sample):
    """
    Args:
      sample (torch.Tensor) - image to be transformed.
      Has shape `(num_channels, height, width)`.
    Returns:
      output (List(torch.Tensor)) - A list of 8 tensors containing the different
      flips and rotations of the original image. Each tensor has the same size as 
      the original image, possibly transposed in the spatial dimensions.
    """
    # BEGIN SOLUTION
    raise NotImplementedError
    # END SOLUTION


##########################################################
# Transforms Compositions
##########################################################
def inference_trans():
  """transforms used for evaluation. Simply convert the images to tensors.
  Returns:
    output (callable) - A transformation that recieves a PIL images and converts
    it to torch.Tensor.
  """
  # BEGIN SOLUTION
  return transforms.ToTensor()
  # END SOLUTION

def default_trans(random_crop_size):
  """transforms used in the basic case for training.
  Args:
    random_crop_size (int / tuple(int, int)) - crop size.
    if int, takes crops of size (random_crop_size x random_crop_size)    
    if tuple, takes crops of size (random_crop_size[0] x random_crop_size[1])
  Returns:
    output (callable) - A transformation that recieves a PIL image, converts it 
    to torch.Tensor and takes a random crop of it. The result's shape is 
    C x random_crop_size x random_crop_size.
  """
  # BEGIN SOLUTION
  if isinstance(random_crop_size, int):
    crop_size = (random_crop_size, random_crop_size)
  else:
    crop_size = random_crop_size

  return transforms.Compose([
    transforms.RandomCrop(crop_size),
    transforms.ToTensor()
  ])
  # END SOLUTION


def advanced_trans(random_crop_size):
    """transforms used in the advanced case for training.
    Args:
      random_crop_size (int / tuple(int, int)) - crop size.
      if int, takes crops of size (random_crop_size x random_crop_size)    
      if tuple, takes crops of size (random_crop_size[0] x random_crop_size[1])
    Returns:
      output (callable) - A transformation that receives a PIL image, converts it 
      to torch.Tensor, takes a random crop of it, and takes the EightCrops of this
      random crop. The result's shape is 8 x C x random_crop_size x random_crop_size.

    Note: you may explore different augmentations for your original implementation.
    """
    # BEGIN SOLUTION
    raise NotImplementedError
    # END SOLUTION


