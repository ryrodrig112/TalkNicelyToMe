import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from gensim.models import KeyedVectors

class PolitenessData(Dataset):
    """
    Dataset class for the politeness data
    Takes in the path of the politeness data, returns dataset in tokenized and embedded but not padded
    """

    def __init__(self, data_path: str, embedding_path: str):
        """
        data_path: path of the data
        embedding_path: str
        """
        data = pd.read_csv(data_path,
                           converters={'all_politeness_scores': pd.eval,
                                       'tokenized_text': pd.eval})
        self.unembedded_docs = data.tokenized_text
        self.politeness_scores = data.avg_politeness_score
        self.lens = data.doc_len
        self.embedding_model = KeyedVectors.load(embedding_path)
        self.embedded_docs = self.embed_data()

    def embed_data(self):
        """
        Embed the unembedded docs using the model KeyedVectors object loaded to the
        """
        doc_vecs = torch.zeros(len(self.unembedded_docs), 76, 300)
        for i in range(len(self.unembedded_docs)):
            doc_vec = torch.zeros(76, 300)  # size = (number of dimension, longest number of words)
            doc = self.unembedded_docs[i]
            for j in range(len(doc)):
                try:
                    if self.embedding_model is None:
                        vec = torch.rand(300)
                    else:
                        vec = self.embedding_model[doc[j]]
                except KeyError:
                    vec = torch.rand(300)
                doc_vec[j, :] += vec
            doc_vecs[i, :, :] += doc_vec
        return doc_vecs

    def __len__(self):
        return len(self.unembedded_docs)

    def __getitem__(self, idx):
        x = {"Unembedded": self.unembedded_docs[idx],
             "Embedded": self.embedded_docs[idx, :, :]}
        y = {"Politeness Score": self.politeness_scores[idx],
             "Length": self.lens[idx]}

        return x, y

