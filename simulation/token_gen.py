#!/usr/bin/env python
'''
Created on 23 Apr 2019

@author: qubix
'''

import sys

from asr.utils import load_data, text_to_features

filename = sys.argv[1]
file_out = sys.argv[2]

print(filename)
data, labels = load_data(filename)
X, word_index = text_to_features(data)

with open(file_out, "w") as f:
    for key in word_index:
        f.write(f"{key}\n")
