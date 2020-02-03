#!/usr/bin/env python

import sys

import asreview
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary


filename = sys.argv[1]
file_out = sys.argv[2]

texts = asreview.ASReviewData.from_file(filename).texts

plain_corpus = [simple_preprocess(text)
                for i, text in enumerate(texts)]

corpus_dict = Dictionary(documents=plain_corpus)

with open(file_out, "w") as f:
    for key in corpus_dict.token2id:
        f.write(f"{key}\n")
