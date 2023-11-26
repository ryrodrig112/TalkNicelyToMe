import torch
import torch.nn as nn

from .encoder import EncoderBasic
from .decoder import DecoderBasic

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.encoder = EncoderBasic()
        self.decoder = DecoderBasic()
   
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar) # do we need this? 
        return self.decode(z, c), mu, logvar