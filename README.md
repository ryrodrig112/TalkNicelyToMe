# TalkNicelyToMe

This repository contains the code for training and evaluating a conditional variational autoencoder (CVAE) for English text politeness. 


0. Install stuff
    a)  packages from `requirements.txt`
    b)  follow the gensim downloader instructions to download and save `word2vec-google-news-300` to your preferred location (we recommend in a folder called `./model/embeddings/` and calling the model `word2vec-google-news-300.model`)
1. Train the model by running `./train.sh` from the command line. This will start a run called "test_run0", and save your model and a plot of the training loss in a ./model/test_run0/ directory. 
    a) if you need to update the embedding path from step `1b`, then please add the `--embedding_path` command line argument.
    b) If you are training a second model with different parameters, give the run a new name to prevent overwriting your previous model. 
    c) Use `python main.py --help` to learn more about parameters you can change.
2. Evaulate the model by running `./evals.sh` from the command line. This will evaluate the best validation model from "test_run0"
    a) if you need to update anything, again please use `python evals.py --help`


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
- `embedded_docs[idx]`
- `politenes_scores[idx]`
- `lens[idx]`

**data CSV fields**
These are the fields in the data CSVs.
- original_text
- tokenized_text
- avg_politeness_score
- politeness_std
- all_politeness_scores
- binary_politeness_score
- doc_len


