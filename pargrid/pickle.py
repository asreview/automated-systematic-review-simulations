"""
Helper scripts to manipulate and create pickle files from embeddings
and data files. Has both an CLI and API.
"""

import os
import pickle
import argparse
import sys

import pandas

from asr.utils import load_data, text_to_features
from asr.models.embedding import load_embedding, sample_embedding


def pickle_dir(create=True):
    """ Returns/creates the directory for the pickle files. """
    pdir = "pickle"
    if create:
        os.makedirs(pdir, exist_ok=True)
    return pdir


def pickle_file(data_file, num_words=20000, create_dir=False):
    """ Get the name for a pickle file, from original file + words """
    pdir = pickle_dir(create_dir)
    bname = os.path.basename(data_file)
    base_name, _ = os.path.splitext(bname)
    suffix = f"_words_{num_words}.pkl"
    pfile = os.path.join(pdir, base_name+suffix)
    return pfile


def parse_arguments(args):
    """ Parse the arguments for writing pickle files. """
    parser = argparse.ArgumentParser(
        description="Creation of pickle files from embeddings + abstracts"
    )
    parser.add_argument(
        "--num_words",
        default=20000,
        type=int,
        help="The number of words."
    )
    parser.add_argument(
        "data_file",
        type=str,
        default=None,
        help="Data file picklify."
    )
    parser.add_argument(
        "embedding_file",
        type=str,
        default=None,
        help="Embedding file."
    )
    return vars(parser.parse_args(args))


def read_pickle(pickle_file):
    return pickle.load(pickle_file)


def write_pickle(data_file, embedding_file, num_words=20000):
    """ Write a pickle file from the embedding + data file. """

    pickle_fp = pickle_file(data_file, num_words, create_dir=True)
    if os.path.isfile(pickle_fp):
        print(f"Pickle file {pickle_fp} already exists, using that.")
        return pickle_fp

#   Load data
    try:
        data = pandas.read_csv(data_file)
    except UnicodeDecodeError:
        data = pandas.read_csv(data_file, encoding="ISO-8859-1")

    texts, y = load_data(data_file)

    # Create features and labels
    print(f"Convert text to features with {num_words} words")
    X, word_index = text_to_features(texts, num_words=num_words)

    # Load embedding
    embedding = load_embedding(embedding_file, word_index=word_index, n_jobs=3)
    embedding_matrix = sample_embedding(embedding, word_index)

    # Write to pickle file.
    with open(pickle_fp, 'wb') as f:
        t = (X, y, embedding_matrix, data)
        pickle.dump(t, f)
    return pickle_fp


def main():
    args = parse_arguments(sys.argv[1:])
    write_pickle(**args)
