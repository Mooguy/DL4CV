import torch
import torch.nn as nn

__all__ = ["weights_init_normal", "Generator", "Discriminator"]

#################################################
# PROVIDED: Init weights
#################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


#################################################
# PROVIDED: Generator
#################################################


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()

        # BEGIN SOLUTION
        self.conv1 = nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False)
        # END SOLUTION

    def forward(self, input):
        """Computes the forward function of the network.

        Args:
         input (torch.Tensor): The input tensor, has shape of `(batch_size, latent_dim, 1, 1)`.

        Returns:
          y (torch.Tensor): The output tensor, has shape of `(batch_size, channels, img_size, img_size)`.
        """
        # BEGIN SOLUTION
        relu = nn.ReLU()
        tanh = nn.Tanh()
        y = relu(self.bn1(self.conv1(input)))
        y = relu(self.bn2(self.conv2(y)))
        y = relu(self.bn3(self.conv3(y)))
        y = tanh(self.conv4(y))
        return y
        # END SOLUTION


#################################################
# IMPLEMENT: Discriminator
#################################################


class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()
        # BEGIN SOLUTION
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        # END SOLUTION

    def forward(self, input):
        """Computes the forward function of the network.

        Args:
         input (torch.Tensor): The input tensor, has shape of `(batch_size, channels, img_size, img_size)`.

        Returns:
          y (torch.Tensor): The output tensor, has shape of `(batch_size)`.
        """
        # BEGIN SOLUTION
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.sigmoid = nn.Sigmoid()
        y = self.leaky_relu(self.conv1(input))
        y = self.leaky_relu(self.bn2(self.conv2(y)))
        y = self.leaky_relu(self.bn3(self.conv3(y)))
        y = self.sigmoid(self.conv4(y))
        return y.squeeze()
        # END SOLUTION
