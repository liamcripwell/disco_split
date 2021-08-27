import re
import string

import nltk
import pandas as pd

from disco_split.processing.connectives import PATTERNS

def reduce_string(s):
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()

def sample_by_connective(input_file, n, random_state=1):
    df = pd.read_csv(input_file)
    sample = df.groupby(["connective"]).sample(n=n, random_state=random_state, replace=True)
    return sample.drop_duplicates()

def check_sent_to_pair(complex_file, simple_file, with_connective=True):
    complex_f = open(complex_file, "r")
    simple_f = open(simple_file, "r")

    hits = []
    while True:
        complex_line = complex_f.readline()
        simple_line = simple_f.readline()
        if complex_line == "":
            break

        complex_sents = nltk.sent_tokenize(complex_line)
        simple_sents = nltk.sent_tokenize(simple_line)

        hit = None
        if len(complex_sents) == 1 and len(simple_sents) == 2:
            if with_connective:
                for r, rel in PATTERNS.items():
                    if hit is not None: break
                    for s, sense in rel.items():
                        if hit is not None: break
                        for connective in sense:
                            pattern = sense[connective]
                            matched = re.search(pattern, reduce_string(simple_sents[-1])) is not None
                            if matched:
                                hit = (r, s, connective, complex_sents, simple_sents)
                                print(hit)
                                break
            else:
                hit = (complex_sents, simple_sents)
            if hit is not None:
                hits.append(hit)

    return hits