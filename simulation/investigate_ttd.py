#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd

from asreview.analysis import Analysis
from asreview.readers import ASReviewData

from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary


def find_unique_words(plain_corpus, doc_freq):
    n_unique = []
    for doc_idx in range(len(plain_corpus)):
        cur_unique = 0
        for word in plain_corpus[doc_idx]:
            if doc_freq[word] == 1:
                cur_unique += 1
        n_unique.append(cur_unique)

    n_words = [len(x) for x in plain_corpus]
    return np.array(n_unique, dtype=int), np.array(n_words, dtype=int)


def count_numbers(texts):
    n_numbers = []
    for text in texts:
        n_numbers.append(sum(c.isdigit() for c in text))
    return np.array(n_numbers, dtype=int)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    data_fp = sys.argv[2]

    analysis = Analysis.from_dir(data_dir)
    ttd = analysis.avg_time_to_discovery()
    ttd_order = sorted(ttd, key=lambda x: ttd[x])
#     for idx in ttd_order:
#         print(f"{idx}: {ttd[idx]}")

    as_data = ASReviewData.from_file(data_fp)
    n_abstract_missing = 0
    n_missing_included = 0
    for i, abstract in enumerate(as_data.abstract):
        if len(abstract) < 10:
            n_abstract_missing += 1
            if as_data.labels[i] == 1:
                n_missing_included += 1

    n_paper = len(as_data.abstract)
    n_included = np.sum(as_data.labels)
    print(f"Number of abstracts missing: {n_abstract_missing}/{n_paper}")
    print(f"Number of included abstracts missing: {n_missing_included}/{n_included}")
    for idx in ttd_order[-10:]:
        as_data.print_record(idx)

    _, texts, _ = as_data.get_data()
    plain_corpus = [simple_preprocess(text)
                    for i, text in enumerate(texts)]
    corpus_dict = Dictionary(documents=plain_corpus)
#     dfs = corpus_dict.dfs
#     ids = corpus_dict.token2id
    doc_freq = {}
    for word, token_id in corpus_dict.token2id.items():
        doc_freq[word] = corpus_dict.dfs[token_id]

    n_unique, n_words = find_unique_words(plain_corpus, doc_freq)
    n_numbers = count_numbers(texts)
    df = pd.DataFrame({
        "n_unique": n_unique[ttd_order],
        "n_words": n_words[ttd_order],
        "unique_freq": n_unique[ttd_order]/n_words[ttd_order],
        "avg_rank": [ttd[x] for x in ttd_order],
        "n_numbers": n_numbers[ttd_order]
    })
    pd.options.display.max_rows = 999
    print(df)
