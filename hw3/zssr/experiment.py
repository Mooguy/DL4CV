from pathlib import Path
import pandas as pd
import utils
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_handling import BasicZSSRDataset, OriginalZSSRDataset, default_trans, inference_trans
from data_handling import advanced_trans, EightCrops
from models import ZSSRNet, ZSSRResNet, ZSSROriginalNet


##########################################################
# Experiment
##########################################################
class Experiment:
  """Trains and evaluates your ZSSR framework on multiple images in a dataset.
  Produces PSNR results per image and total average PSNR. You may use the 
  different components (both basic and advanced) you have implemented so far.
  Eventually come up with your best configuration, training and evaluation
  method.
  
  NOTE: You may add more members to this class as long as you don't change the
  original members.
  """

  def __init__(self, dataset_root, config):
    """
    Args:
      dataset_root (pathlib.Path): Path to the root of the dataset.
      config (dict): Configuration dictionary. 
      Contains parameters from training & evaluation.
    """    
    self.train_paths = sorted([f for f in (dataset_root / 'train').glob('*.png')])
    self.gt_paths = sorted([f for f in (dataset_root / 'gt').glob('*.png')])
    # BEGIN SOLUTION
    self.dataset_root = dataset_root
    for key, value in config.items():
      setattr(self, key, value)
    # END SOLUTION

  def baseline(self, image_path):
    """ Finds the baseline PSNR for a specific image, 
    when using bicubic interpolation (no learning).

    Args:
      image_path (pathlib.Path): Path to the image to be evaluated.
    Returns:
      image_psnr (float): The PSNR value between the original image and the 
      image created with bicubic interpolation.
    """
    # define dataset
    dataset = BasicZSSRDataset(image_path, self.scale_factor, inference_trans())
    # fetch an instance
    inputs = dataset[0]
    # parse it to gt and input
    gt_image, resize_image = inputs['SR'][None, ...], inputs['LR'][None, ...]
    # resize with bicubic interpolation
    bicubic_image = utils.rr_resize(resize_image, scale_factors=2)
    # compute baseline psnr
    baseline_psnr = utils.psnr(bicubic_image[0], gt_image[0])
    return baseline_psnr.item()

  def train(self, image_path):
    """ Trains a ZSSR model on a specific image.
    Args:
      image_path (pathlib.Path): Path to the image to be trained.

    Returns:
      model (torch.nn.Module): A trained model.

    We hinted specific steps and parts of the training loop and visualization.
    You may change these if you prefer.
    """
    # BEGIN SOLUTION
    
    # define dataset
    training_dataset = BasicZSSRDataset(image_path, self.scale_factor, default_trans(self.random_crop_size))
    
    # define model
    model = ZSSRNet(self.scale_factor).to(self.device)
    
    # define loss
    loss_function = torch.nn.L1Loss()
    
    # define optimizer & schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.95)
    
    # define visualizer
    if self.verbose:
        visualizer = utils.Visualizer()
        accumulated_loss, accumulated_psnr = 0, 0

    # training loop
    for epoch in range(1, self.epochs + 1):
        data_instance = training_dataset[0]
        ground_truth_image = data_instance['SR'].unsqueeze(0).to(self.device)
        low_res_image = data_instance['LR'].unsqueeze(0).to(self.device)
        
        # forward pass
        predictions = model(low_res_image)
        
        # calculate the loss
        loss = loss_function(predictions, ground_truth_image)
        
        # zero gradients
        optimizer.zero_grad()
        
        # backward pass
        loss.backward()
        
        # do optimization step
        optimizer.step()
        
        # step the scheduler
        scheduler.step()
        
        # log visualizations if necessary.
        if self.verbose:
            accumulated_loss += loss.item()
            accumulated_psnr += utils.psnr(predictions[0], ground_truth_image[0])      
            if epoch % self.show_interval == 0: 
                validation_psnr = self.evaluate(image_path, model)  
                visualizer.update(accumulated_loss / self.show_interval, 
                                  accumulated_psnr / self.show_interval,
                                  validation_psnr)
                accumulated_loss, accumulated_psnr = 0, 0

    return model
    # END SOLUTION

  @torch.no_grad()
  def evaluate(self, image_path, model):
      """ Evaluates a ZSSR model on a specific image.
      Args:
        image_path (pathlib.Path): Path to the image to be trained.
        model (torch.nn.Module): A trained model.
      Returns:
        image_psnr (float): The PSNR value between the original image and the 
        image created with the model.
      """
      # BEGIN SOLUTION
      # define dataset
      evaluation_dataset = BasicZSSRDataset(image_path, self.scale_factor, inference_trans())
      
      # fetch an instance
      data_instance = evaluation_dataset[0]
      ground_truth_image = data_instance['SR'].unsqueeze(0).to(self.device)
      low_res_image = data_instance['LR'].unsqueeze(0).to(self.device)
      
      # forward pass
      with torch.no_grad():
          model.eval()
          predictions = model(low_res_image)
      
      # calculate PSNR
      image_psnr = utils.psnr(predictions[0], ground_truth_image[0])
      
      return image_psnr.item()
      # END SOLUTION

  def run(self):
    """ Run an entire experiment.
    Returns:
      run_df (pd.DataFrame): dataframe of all PSNR values.
    """
    run_list = []
    avg_psnr, avg_baseline_psnr = 0, 0

    # train and evaluate every image
    for train_path, gt_path in tqdm(zip(self.train_paths, self.gt_paths)):
      # make sure train and gt images match
      assert train_path.name == gt_path.name
      print(train_path.name)
      # compute baseline PSNR for comparison
      baseline_psnr = self.baseline(gt_path)
      avg_baseline_psnr += baseline_psnr

      model = self.train(train_path)
      psnr = self.evaluate(gt_path, model)
      run_list.append({'image_path': str(gt_path.name), 
                       'psnr': psnr,
                       'baseline': baseline_psnr
                      })
      avg_psnr += psnr

    # compute average psnr
    avg_psnr = avg_psnr / len(self.gt_paths)
    avg_baseline_psnr = avg_baseline_psnr / len(self.gt_paths)
    run_list.append({'image_path': 'TOTAL_AVG', 
                     'psnr': avg_psnr, 
                     'baseline': avg_baseline_psnr})
    
    # create results file
    run_df = pd.DataFrame(run_list)
    run_df.to_csv(utils.ROOT / f'{self.dataset_root.name}_psnr.csv')
    return run_df


