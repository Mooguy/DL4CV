class Module:

  def __init__(self):
    self._parameters = []
    self._buffers = []
    self._modules = []
    self._train = True

  def forward(self, *args, ctx=None):

    raise NotImplementedError  # DO NOT EDIT THIS. IT JUST MEANS THIS METHOD IS ABSTRACT.

  def __call__(self, *args, **kwargs):

    return self.forward(*args, **kwargs)

  def to(self, device):

    device = torch.device(device)
    # transfer all the parameters to device
    for param in self._parameters:
      setattr(self, param, getattr(self, param).to(device=device))
    # transfer all the buffers to device
    for buffer in self._buffers:
      setattr(self, buffer, getattr(self, buffer).to(device=device))
    # transfer all the modules to device
    for module in self._modules:
      setattr(self, module, getattr(self, module).to(device=device))
    return self

  def parameters(self, recurse=True):

    # add all parameters
    parameters = []
    parameters += [getattr(self, param) for param in self._parameters]
    if recurse:
      # add all parameters of sub-modules
      for module in self._modules:
        parameters += getattr(self, module).parameters(recurse=True)
    # deduplicate
    return list(set(parameters))

  def train(self, recurse=True):

    self._train = True
    if recurse:
      for module in self._modules:
        getattr(self, module).train(recurse=True)

  def eval(self, recurse=True):

    self._train = False
    if recurse:
      for module in self._modules:
        getattr(self, module).eval(recurse=True)

class Linear(Module):

  def __init__(self, in_dim, out_dim):

    super().__init__()
    # BEGIN SOLUTION
    self.weight = torch.empty(out_dim, in_dim)
    
    # Create bias parameter with shape (out_dim,)
    self.bias = torch.empty(out_dim)

    # Add parameter names to self._parameters
    self._parameters.extend(['weight', 'bias'])
    
    # Initialize parameters
    self.init_parameters()
    # END SOLUTION

  def init_parameters(self):

    # BEGIN SOLUTION
    self.weight.data.normal_(0, 0.01)

    self.bias.data.fill_(0)
    # END SOLUTION

  def forward(self, x, ctx=None):

    # BEGIN SOLUTION
    if self.weight.device != x.device:
        self.weight = self.weight.to(x.device)
    if self.bias.device != x.device:
        self.bias = self.bias.to(x.device)
    y = linear(x, self.weight, self.bias, ctx=ctx)
    return y
    # END SOLUTION
class SoftmaxClassifier(Module):

  def __init__(self, in_dim, num_classes):
    super().__init__()
    # BEGIN SOLUTION
    self.fc = Linear(in_dim, num_classes)
    # END SOLUTION

  def forward(self, x, ctx=None):

    # BEGIN SOLUTION
    x = x.view(x.size(0), -1)
    y = self.fc(x)
    return y
    # END SOLUTION