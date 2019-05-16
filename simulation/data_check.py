import pickle
import numpy as np

with open("pickle/Depression Cuipers et al. 2018_words_20000_+0.pkl", "rb") as f:
    data = pickle.load(f)
labels = data[1]

for label in labels:
    try:
        int(label)
    except ValueError:
        print(label)
with open("label_test.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")
