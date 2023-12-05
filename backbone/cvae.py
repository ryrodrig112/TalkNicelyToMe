import torch
import torch.nn as nn

from .encoder import EncoderBasic
from .decoder import DecoderBasic

class CVAE(nn.Module):
    def __init__(self, embedding_size = 300, feature_size = 76, latent_size = 300, class_size = 25):
        super(CVAE, self).__init__()

        self.encoder = EncoderBasic(
            embedding_size,
            feature_size,
            latent_size,
            class_size,
            )

        self.decoder = DecoderBasic(
            feature_size,
            latent_size,
            class_size,
            )
   
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar) # do we need this? 
        return self.decoder(z, c), mu, logvar
        