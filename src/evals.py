import os
import argparse
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from backbone import cvae
from src.utils import *
from gensim.models import KeyedVectors
import numpy as np

# set up torch
random_seed= 42
batch_size = 16
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# set load and save paths
train_data = '../data/train_data.csv'
test_data = '../data/test_data.csv'
model_dir = "../models/cvae/test_run0"
model_params_file = 'epoch_29_27.904600340138924_27.380401288761814.pth'
embedding_dir = '../models/embeddings/'
embedding_file = "word2vec-google-news-300.model"
results_dir = "../data/eval_results"

# create datasets and dataloaders
print("Loading Data")
dataset = PolitenessData(test_data, os.path.join(embedding_dir, embedding_file))
loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

# init and load CVAE
print("Loading Models")
model = cvae.CVAE(class_size = 1)
model.load_state_dict(torch.load(os.path.join(model_dir, model_params_file)))
embedding_model = KeyedVectors.load(os.path.join(embedding_dir, embedding_file))
file = open("../data/google-10000-english.txt", "r")
data = file.read()
vocab = data.split("\n")
file.close()
filtered_embedding_model = KeyedVectors(vector_size=embedding_model.vector_size)
for word in vocab:
    if word in embedding_model:
        vector = embedding_model[word]
        filtered_embedding_model.add_vector(word, vector)

print("Inference")
for i, (x, c, _) in enumerate(loader):
    n = x.shape[0]
    c = c.unsqueeze_(1).to(torch.float32)
    x_flat = x.flatten(start_dim=1)
    #   Encode samples into latent space
    mu_transform, logvar_trasnform = model.encoder(x_flat, c)
    z_transform = model.reparameterize(mu_transform, logvar_trasnform)

    # decode into new politeness
    c_new_size = c.size()
    c_new_tone = torch.normal(mean=14.09, std=3.124, size=c_new_size)  # uniform distribution from 1-25
    x_new_tone = model.decoder(z_transform, c_new_tone)

    # send the politness transformed version back into the encoder
    mu_recover, logvar_recover = model.encoder(x_new_tone, c_new_tone)
    z_recover = model.reparameterize(mu_recover, logvar_recover)

    # decode to original politeness
    x_recon = model.decoder(z_recover, c)
    x_recon = x_recon.reshape(n, 76, 300)
    if i > 2:
        break

x_recon = x_recon.detach().numpy()

if not os.path.exists(results_dir):
   os.makedirs(results_dir)

# Calculate L2 Distances
print("Calculating L2 Distance")
squared_error = (dataset[:][0].numpy() - x_recon)**2
l2_distances = np.mean(squared_error, axis=(1, 2))

# Calculate Edit Distances
print("Calculating Edit Difference")
test_df = pd.read_csv(test_data,  converters={'all_politeness_scores': pd.eval,
                                       'tokenized_text': pd.eval})
docs = list(test_df.tokenized_text)
reconstructed_docs = embedding_vectors_to_words(x_recon, filtered_embedding_model)

edit_distances = []
for i in range(len(docs)):
    edit_distance_ratio = calc_edit_distance_ratio(docs[i], docs[i])
    edit_distances.append(edit_distance_ratio)

#Save Quantitative Evaluation
quantitative_results_file = "eval.csv"
reconstructed_embedding_file = "reconstructed_embeddings.npy"
original_embedding_file = "original_embeddings.npy"


with open(os.path.join(results_dir, reconstructed_embedding_file), 'wb') as f:
    np.save(f, x_recon)

with open(os.path.join(results_dir, original_embedding_file), 'wb') as f:
    np.save(f, x)


data_dict = {"original_doc": [" ".join(docs[i]) for i in range(len(docs))],
             'reconstructed_doc': [" ".join(reconstructed_docs[i]) for i in range(len(reconstructed_docs))],
             'token_edit_distance': edit_distances,
             'embedding_l2_distance': list(l2_distances)}
results_df = pd.DataFrame(data_dict)
results_df.to_csv(os.path.join(results_dir, quantitative_results_file), index=False)

#Novel Sentences
print("Generating Novel Sentence")
z_novel = torch.normal(mean=0, std=1, size=(100, 300))
c_novel = torch.rand(size=(100, 1))*25
embeddings_novel = model.decoder(z_novel, c_novel)
embeddings_novel = embeddings_novel.reshape(100, 76, 300).detach().numpy()
novel_docs = embedding_vectors_to_words(embeddings_novel, filtered_embedding_model)
novel_doc_as_str = [" ".join(novel_docs[i]) for i in range(len(novel_docs))]
novel_sentences_file = "novel_sentences.csv"
novel_dict = {"sentence": novel_doc_as_str,
              "politeness": list(c_novel.detach().numpy())}
novel_sentences_df = pd.DataFrame(novel_dict)
novel_sentences_df.to_csv(os.path.join(results_dir, novel_sentences_file), index=False)