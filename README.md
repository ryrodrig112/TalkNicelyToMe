# TalkNicelyToMe

### Overview of PolitenessData Dataset Class
**Constructor Parameters**:
- `data_path`: path to the dataset to load
- `embedding_path:` path to the saved key/vector embeddings from a pretrained gensim model

**Attributes:**
- `unembedded_docs`: pd.Series of the documents from the Wiki and Stack Politeness datasets 
- `politenes_scores`: pd.Series containing the average politeness score for each document
- `lens`: pd.Series with the length of each document
- `embedding_model`: loads the key/vector dictionary specified in `embedding_path`
- `embedded_docs`: Embeds the documents from `unembedded_docs` into a tensor of size (n, 76, 300)
    - n: number of documents
    - 76: maximum length of a document
    - 300: dimension of an individual embedding

**__get_item__ method**
parameter: 
- `idx`: integer within 0 to 1-n
returns: 
- x: (`unembedded_docs[idx], embedded_docs[idx]`)
- y: (`politenes_scores[idx], lens[idx]`)


