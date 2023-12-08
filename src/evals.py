import os
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from backbone import cvae
from src.utils import PolitenessData


#need util functions for:
# 0. Stratify out test set
# 1. finding closest word to point
    # your_word_vector = array([-0.00449447, -0.00310097, 0.02421786, ...], dtype=float32)
    # model.most_similar(positive=[your_word_vector], topn=1))
# 2. token edit distance
    # nltk.edit_distance
# 3. summed l2d embedding distance
    # me write
# 4. clipping predictions to length?

# argparse inputs
num_epochs = 100
lr = 0.001
batch_size = 16
random_seed= 42
run_name = 'test_run0'

# set up torch
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# make our model directory
model_dir = os.path.join('.', 'models', run_name)
os.makedirs(model_dir, exist_ok=True)

# load data
data_path = './data/fil_politeness_data.csv'
embedding_path = './models/word2vec-google-news-300.model'

# create datasets and dataloaders
dataset = PolitenessData(data_path, embedding_path)
validation_split = .2

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# init and load CVAE
model = cvae.CVAE(class_size = 1)
model.load_state_dict(torch.load(PATH))

optimizer = torch.optim.SGD(
    params = model.parameters(),
    lr = lr,
    momentum=0.8,
    )

# init scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=60,
    gamma=0.5,
    )

#load trained model parameters


# Quantitative Evaluation:
# Reconstruction Ability For: (comment (politeness: A) -> transformed comment (Politeness: B) -> reconstructed comment politeness (A)
for i, (x, c, _) in enumerate(val_dl):
    # send it into the encoder
    c_new  = ... # uniform distribution from 1-25
    mu, logvar = model.encoder(x, c)
    z = model.reparameterize(mu, logvar)
    out = model.decoder(z, c_new)

    mu_new, logvar_new = model.encoder(out, c_new)
    z_prime = model.reparameterize(mu_new, logvar_new)
    x_prime = model.decoder(z, c_new)

    token_edit_distance = ...
    summed_l2_distance = ...


# Quantitative Evaluation:
# Naturalness and politeness of evaluated texts
    #: sampling randomly
    #: trying a linear walk walk
