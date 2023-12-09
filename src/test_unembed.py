import os
import torch
from gensim.models import KeyedVectors

from src.utils import PolitenessData

# argparse input
batch_size = 16
random_seed= 42
run_name = 'test_run0'

# set up torch
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# load data 
data_path = './data/fil_politeness_data.csv'
embedding_path = './models/word2vec-google-news-300.model'
model_path = '' # TODO: put ur path here

# create datasets and dataloaders
dataset = PolitenessData(data_path, embedding_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle = True)

# init cvae 
model = torch.load(model_path)

# test separately
wv = KeyedVectors.load(embedding_path)
model.eval()
with open(os.path.join(os.path.split(model_path)[0], "val_recon_out.txt"), "w") as f:
    print(f)
    with torch.no_grad():
        for i, (x, c, _)  in enumerate(val_dl):
            actual_bs = x.shape[0]
            c = c.unsqueeze_(1).to(torch.float32)
            x = x.flatten(start_dim = 1)
        
            output, mu, logvar = model(x, c)

            output = output.reshape((actual_bs, 76, 300)).detach().numpy()
            x = x.reshape((actual_bs, 76, 300)).detach().numpy()
    
            for j in range(actual_bs):
                print(i, j)
                recon_doc = []
                doc = []
                for k in range(76):
                    doc.append(wv.most_similar(x[j,k], topn=1)[0][0])
                    recon_doc.append(wv.most_similar(output[j,k], topn=1)[0][0])
                f.write(' '.join(doc))
                f.write(' '.join(recon_doc))