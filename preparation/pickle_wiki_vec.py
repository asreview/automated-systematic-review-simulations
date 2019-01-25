import os
import pickle

from asr.models.embedding import load_embedding

# Load embedding layer.
embedding_fp = os.path.join('data', 'pretrained_models', 'wiki.en.vec')
embedding = load_embedding(embedding_fp)

# Write to pickle file.
if not os.path.exists('pickle'):
    os.makedirs('pickle')

# Dump the result to a pickle file.
#
# Due to an issue, this gives problems on MacOS systems.
# Check out check issue https://stackoverflow.com/questions/31468117
#
# The following code works on Windows and Linux
#
# pickle_fp = os.path.join('pickle', 'word2vec_wiki_en.pkl')
# with open(pickle_fp, 'wb') as f:
#     pickle.dump(embedding, f)
pickle_fp = os.path.join('pickle', 'word2vec_wiki_en.pkl')
bytes_out = pickle.dumps(embedding)
with open(pickle_fp, 'wb') as f_out:
    for idx in range(0, len(bytes_out), 2**31 - 1):
        f_out.write(bytes_out[idx:idx + 2**31 - 1])