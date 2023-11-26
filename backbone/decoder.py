import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class DecoderBasic(nn.Module):
    # modified from https://github.com/unnir/cVAE/blob/master/cvae.py
    def __init__(self, embedding, feature_size, latent_size, class_size):
        super(DecoderBasic, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.feature_size = feature_size 
        self.latent_size = latent_size
        self.class_size = class_size

        # Create word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding) 

        #layers
        self.dec0 = nn.Linear(latent_size + class_size, 400)
        self.dec1 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, , c):
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.dec0(inputs))
        h4 = self.dec1(h3)

        return self.sigmoid(h4), h4

class DecoderLSTM(nn.Module):
    # COPIED COMPLETELY FROM https://github.com/artidoro/conditional-vae
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0):
        super(DecoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        
        # Create word embedding, LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding) 
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)
    
    def forward(self, h):
        # Embed text and pass through GRU
        x = self.embedding(x)
        x = self.dropout(x)

        # Create initial hidden state of zeros: 2-tuple of num_layers x batch size x hidden dim
        num_layers = self.num_layers
        init = Variable(torch.zeros(num_layers, x.size(1), self.h_dim), requires_grad=False)
        init = init.cuda() if use_gpu else init
        h0 = (init, init.clone())

        out, h = self.lstm(x, h0)
        return out, h