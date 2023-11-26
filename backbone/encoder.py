import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class EncoderBasic(nn.Module):
    # modified from https://github.com/unnir/cVAE/blob/master/cvae.py
    def __init__(self, embedding, feature_size, latent_size, class_size):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.class_size = class_size

        # Create word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding)

        # Layers
        self.enc0  = nn.Linear(self.embedding_size * feature_size + class_size, 400)
        self.enc11 = nn.Linear(400, latent_size)
        self.enc12 = nn.Linear(400, latent_size)

        self.elu = nn.ELU()
    
    def forward(self, x, c): # Q(z|x, c)
        '''
        #TODO: determine how to fix feature size (fix each sentence to 20 words? fill with OOV? and truncate? not great) 
        x: (bs, feature size)  
        c: (bs, class_size)
        '''
        x = self.embedding(x)

        inputs = torch.cat([x, c], 1) # (bs, feature_size*embedding_size+class_size)
        h1 = self.elu(self.enc0(inputs))
        out = self.enc11(h1)
        z = self.enc12(h1)
        return out, z



class EncoderLSTM(nn.Module):
    # COPIED COMPLETELY FROM https://github.com/artidoro/conditional-vae
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0, bidirectional=True):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers = num_layers 
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        # Create word embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding)
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        
        # Embed text 
        x = self.embedding(x)
        x = self.dropout(x)

        # Create initial hidden state of zeros: 2-tuple of num_layers x batch size x hidden dim
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        init = Variable(torch.zeros(num_layers, x.size(1), self.h_dim), requires_grad=False)
        init = init.cuda() if use_gpu else init
        h0 = (init, init.clone())

        # Pass through LSTM
        out, h = self.lstm(x, h0) # maybe have to pad now?
        return out, h
