import os
import argparse
import numpy as np

import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F

from backbone import cvae 
from src.utils import PolitenessData

# argparse inputs
num_epochs = 30
lr = 3e-8
batch_size = 16
random_seed= 42
run_name = 'test_run3'

# set up torch
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# make our model directory
model_dir = os.path.join('.', 'models', run_name)
os.makedirs(model_dir, exist_ok=True)

# load data 
data_path = './data/train_data.csv'
embedding_path = './models/embeddings/word2vec-google-news-300.model'

# create datasets and dataloaders
dataset = PolitenessData(data_path, embedding_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle = True)

# init cvae 
model = cvae.CVAE(class_size = 1) # class_size = 25 if we round then one_hot

# init optimizer
optimizer = torch.optim.SGD(
    params = model.parameters(), 
    lr = lr, 
    momentum=0.8,
    )
optimizer.zero_grad()

# init scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=10, 
    gamma=0.5,
    )

# define loss function
def loss_function(data, recon_x, mu, logvar, beta = 0.001):
    """Computes the loss = -ELBO = Negative Log-Likelihood + KL Divergence.
    
    Args: 
        recon_x: Decoder output.
        data: Ground truth outpu
        label:
        mu: Mean of Z
        logvar: Log-Variance of Z
        
        p(z) here is the standard normal distribution with mean 0 and identity covariance.
    """
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL Divergence b/w q_\phi(z|x) || p(z)

    MAE = F.l1_loss(recon_x, data, reduction='sum')
    return 0.001*MAE + KLD
    # MSE = F.mse_loss(recon_x, data, reduction='sum')
    # return MSE + beta * KLD

    # BCE = F.binary_cross_entropy(recon_x, data, reduction='sum') # BCE = -Negative Log-likelihood
    # return BCE + KLD

tr_losses = []
val_losses = []
best_val_loss = np.inf
for epoch in range(num_epochs): 
    print(f"epoch {epoch}")
    ## train
    model.train()
    tr_loss_e = []
    for i, (x, c, _) in enumerate(train_dl):
        actual_bs = x.shape[0]
        c = c.unsqueeze_(1).to(torch.float32)
        # c = torch.nn.functional.one_hot(torch.round(c), num_classes=25)
        x = x.flatten(start_dim = 1)
        output, mu, logvar = model(x, c)
        output.reshape((actual_bs, 76, 300))

        loss = loss_function(x, output, mu, logvar)
        tr_loss_e.append(loss.item())
        loss.backward()
        optimizer.step()

    scheduler.step()
    tr_losses.append(np.mean(tr_loss_e))
    print(f"train loss {tr_losses[-1]}")

    ## validation
    model.eval()
    val_loss_e = []
    with torch.no_grad():
        for i, (x, c, _)  in enumerate(val_dl):
            actual_bs = x.shape[0]
            c = c.unsqueeze_(1).to(torch.float32)
            # c = torch.nn.functional.one_hot(torch.round(c), num_classes=25)
            x = x.flatten(start_dim = 1)
            
            output, mu, logvar = model(x, c)
            output.reshape((actual_bs, 76, 300))

            loss = loss_function(x, output, mu, logvar)
            val_loss_e.append(loss.item())
        val_losses.append(np.mean(val_loss_e))
        print(f"val loss {val_losses[-1]}")

    # plot performance and save model
    plt.plot([l.item() for l in tr_losses], 'b', label='train loss')
    plt.plot([l.item() for l in val_losses], 'r', label='validataion loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, "train_val_loss.png"))
    plt.close()

    # save models
    if val_losses[-1] < best_val_loss: 
        torch.save(model.state_dict(), os.path.join(model_dir, f"best_model.pth"))
        best_val_loss = val_losses[-1]
    if epoch%10 == 9: 
        torch.save(model.state_dict(), os.path.join(model_dir, f"epoch_{epoch}_{tr_losses[-1]}_{val_losses[-1]}.pth"))

# test separately
