import os
import pickle

from keras.utils import to_categorical

from asr.utils import load_data, text_to_features
from asr.models.embedding import load_embedding, sample_embedding

# load data
data_fp = os.path.join('data', 'datasets', 'ptsd_vandeschoot.csv')
data, labels = load_data(data_fp)

# create features and labels
X, word_index = text_to_features(data)
y = to_categorical(labels) if labels.ndim == 1 else labels

# Load embedding layer.
embedding_fp = os.path.join('data', 'pretrained_models', 'wiki.en.vec')
embedding, words = load_embedding(embedding_fp)
embedding_matrix = sample_embedding(embedding, words, word_index)

# Write to pickle file.
if not os.path.exists('pickle'):
    os.makedirs('pickle')

pickle_fp = os.path.join('pickle', 'ptsd_vandeschoot.pkl')
with open(pickle_fp, 'wb') as f:
    pickle.dump((X, y, embedding_matrix), f)
