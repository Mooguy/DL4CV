import torch.nn.functional as F
import torch
import torchvision
from torchvision import transforms
from models import ViT
import time

def train_epoch(model, optimizer, data_loader, loss_history, device):
  total_samples = len(data_loader.dataset)
  model.train()

  for i, (data, target) in enumerate(data_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = F.log_softmax(model(data), dim=1)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
      print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
            ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
            '{:6.4f}'.format(loss.item()))
      loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history, device):
  model.eval()
  
  total_samples = len(data_loader.dataset)
  correct_samples = 0
  total_loss = 0

  with torch.no_grad():
    for data, target in data_loader:
      data = data.to(device)
      target = target.to(device)
      output = F.log_softmax(model(data), dim=1)
      loss = F.nll_loss(output, target, reduction='sum')
      _, pred = torch.max(output, dim=1)
      
      total_loss += loss.item()
      correct_samples += pred.eq(target).sum()

  avg_loss = total_loss / total_samples
  loss_history.append(avg_loss)
  print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
        '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


def train_vit():

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  start_time = time.time()

  BATCH_SIZE_TRAIN = 100
  BATCH_SIZE_TEST = 1000
  N_EPOCHS = 6
  
  DOWNLOAD_PATH = '/home/labs/antebilab/guyilan/Courses/DL4CV/hw4/data/mnist'

  # Load data
  transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

  train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True, transform=transform_mnist)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2)

  test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=2)

  # Initialize model
  model = ViT(28, 7, 1, 10, 64, 6, 8, .02).to(device)
    
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

  # Train model
  train_loss_history, test_loss_history = [], []
  for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    curr_start_time = time.time()
    train_epoch(model, optimizer, train_loader, train_loss_history, device)
    evaluate(model, test_loader, test_loss_history, device)
    print(f'Epoch {epoch} execution time:', '{:5.2f}'.format((time.time() - curr_start_time) / 60), 'minutes\n')

  print('Execution time:', '{:5.2f}'.format((time.time() - start_time) / 60), 'minutes')

