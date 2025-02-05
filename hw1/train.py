import torch  # noqa

from model import softmax_classifier
from model import softmax_classifier_backward
from model import cross_entropy
from utils import Metric, accuracy  # noqa

__all__ = ['create_model', 'test_epoch', 'test_epoch', 'train_loop']


#################################################
# create_model
#################################################

def create_model():
  """
  Creates a Softmax Classifier model `(w, b)`.

  Returns:
      w (torch.Tensor): The weight tensor, shape `(num_classes, in_dim)`.
      b (torch.Tensor): The bias tensor, shape `(num_classes,)`.
  """
  #intialize dimensions of flattened image
  in_dim = 28 * 28 
  #intialize number of classes
  num_classes = 10  

  #initialize weight tensor
  w = torch.randn(num_classes, in_dim) * 0.01  

  #initialize bias tensor with zeros
  b = torch.zeros(num_classes) 

  #torch option to calculate gradients for theses variables during training:
  w.requires_grad = True
  b.requires_grad = True

  return w, b


#################################################
# train_epoch
#################################################

def train_epoch(w, b, lr, loader):
    """
    Trains over an epoch, and returns the accuracy and loss over the epoch.

    Args:
        w (torch.Tensor): The weight tensor, shape `(num_classes, in_dim)`.
        b (torch.Tensor): The bias tensor, shape `(num_classes,)`.
        lr (float): The learning rate.
        loader (torch.utils.data.DataLoader): A data loader for the dataset.

    Returns:
        acc_metric (Metric): The accuracy metric over the epoch.
        loss_metric (Metric): The loss metric over the epoch.
    """
    device = w.device

    #Metric is a class from utils.py that is used to calculate the average of the loss and accuracy
    loss_metric = Metric()
    acc_metric = Metric()

    for x, y in loader:
        x, y = x.to(device=device), y.to(device=device)
        # BEGIN SOLUTION

        #flatten the input tensor
        x = x.view(x.size(0), -1)

        #make predictions using the softmax classifier
        pred = softmax_classifier(x, w, b)
        
        #compute the cross entropy loss
        loss = cross_entropy(pred, y)

        softmax_classifier_backward(x, w, b, pred, y)

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            w.grad.zero_()
            b.grad.zero_()

        acc = accuracy(pred, y)
        # END SOLUTION
        loss_metric.update(loss.item(), x.size(0))
        acc_metric.update(acc.item(), x.size(0))

    return loss_metric, acc_metric


#################################################
# test_epoch
#################################################

def test_epoch(w, b, loader):
  """Evaluating the model at the end of the epoch.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    loader (torch.utils.data.DataLoader): A data loader. An iterator over the dataset.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
  device = w.device

  loss_metric = Metric()
  acc_metric = Metric()
  for x, y in loader:
    x, y = x.to(device=device), y.to(device=device)
    # BEGIN SOLUTION

    x = x.view(x.size(0), -1)

    pred = softmax_classifier(x, w, b)

    loss = cross_entropy(pred, y)

    #dont need to calculate gradients for test data

    acc = accuracy(pred, y)
    # END SOLUTION
    loss_metric.update(loss.item(), x.size(0))
    acc_metric.update(acc.item(), x.size(0))
  return loss_metric, acc_metric


#################################################
# PROVIDED: train
#################################################

def train_loop(w, b, lr, train_loader, test_loader, epochs, test_every=1):
  """Trains the Softmax Classifier model and report the progress.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    lr (float): The learning rate.
    train_loader (torch.utils.data.DataLoader): The training set data loader.
    test_loader (torch.utils.data.DataLoader): The test set data loader.
    epochs (int): Number of training epochs.
    test_every (int): How frequently to report progress on test data.
  """
  for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_epoch(w, b, lr, train_loader)
    print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
          f'Loss: {train_loss.avg:7.4g}',
          f'Accuracy: {train_acc.avg:.3f}',
          sep='   ')
    if epoch % test_every == 0:
      test_loss, test_acc = test_epoch(w, b, test_loader)
      print(' Test', f'Epoch: {epoch:03d} / {epochs:03d}',
            f'Loss: {test_loss.avg:7.4g}',
            f'Accuracy: {test_acc.avg:.3f}',
            sep='   ')
