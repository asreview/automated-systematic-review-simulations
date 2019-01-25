# Automated Systematic Review - Simulation study

This project contains the code of the simulation study for the [Automated
Systematic Review](https://github.com/msdslab/automated-systematic-review)
project. This repository is used to simulate, aggregate and visualise the
active learning process.

We make use of the SURFSara HPC infrastructure. But with some modifications, 
you can run the code on other HPC facility as well.

## Installation 

The Automated Systematic Review project requires Python 3.6+. 

Install the Automated Systematic Review project directly from the github page
https://github.com/msdslab/automated-systematic-review. One can do this with
pip and git. Also install the additional dependencies 

``` bash
pip install -r requirements.txt
pip install git+https://github.com/msdslab/automated-systematic-review.git
```

## Data preparation

See also: [preparation/](preparation/)

The preparation of the data and the embedding of the model takes quite some
computational time. To prevent each node from doing the same computations for
preparation, we store the prepared objects in Python pickle files.

``` bash
python preparation/pickle_wiki_vec.py 
python preparation/pickle_ptsd_vandeschoot.py --words=10000
python preparation/pickle_ptsd_vandeschoot.py --words=20000
```

This generates the following files:
```
pickle/
├── ptsd_vandeschoot_words_10000.pkl
├── ptsd_vandeschoot_words_20000.pkl
└── word2vec_wiki_en.pkl

0 directories, 3 files
```

The following code gives a outline of what the code looks like. 

``` python 

# load dependencies
import pickle

from keras.utils import to_categorical

from asr.utils import load_data, text_to_features
from asr.models.embedding import load_embedding, sample_embedding

# load data
data, labels = load_data(PATH_TO_DATA)

# create features and labels
X, word_index = text_to_features(data)
y = to_categorical(labels) if labels.ndim == 1 else labels

# Load embedding layer. 
embedding, words = load_embedding(PATH_TO_EMBEDDING)
embedding_matrix = sample_embedding(embedding, words, word_index)

# Write to pickle file.
with open(PATH_TO_PICKLE, 'wb') as f:
    pickle.dump((X, y, embedding_matrix), f)
```

In a simulation study, you can quickly load the data stored in the pickle file. 

``` python
with open(PATH_TO_PICKLE, 'rb') as f:
    X, y, embedding_matrix = pickle.load(f)
```

## Simulations

```python

import asr
from asr.model import create_lstm_model

# load the data 
with open(PATH_TO_PICKLE, 'rb') as f:
    X, y, embedding_matrix = pickle.load(f)

# create the model
model = create_lstm_model(
    backwards=True,
    dropout=0.4,
    optimizer='rmsprop',
    embedding_layer=embedding_matrix
)

reviewer = asr.Review(model)
reviewer.oracle(X, y)

# store the review training logs and other metadata
reviewer.logs
```
