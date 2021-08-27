import re
import string

import nltk
import pandas as pd

from disco_split.processing.connectives import PATTERNS

def reduce_string(s):
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()

def find_candidates(input_file, output_file=None):
    df = pd.read_csv(input_file, sep="\t", names=["complex", "simple"])

    candidates = []
    for i, sample in df.iterrows():
        simples = sample.simple.split("<::::>")
        if len(simples) != 2: continue

        for r, rel in PATTERNS.items():
            for s, sense in rel.items():
                for connective in sense:
                    pattern = sense[connective]
                    matched = re.search(pattern, reduce_string(simples[1])) is not None
                    if matched:
                        hit = (input_file, i, r, s, connective, sample.complex, " ".join(simples))
                        candidates.append(hit)
                        print(hit)
                        break

    out_df = pd.DataFrame(candidates, columns=[
            "source", "source_id", "relation", "sense", "connective", 
            "complex", "simple"])

    if output_file is not None:
        out_df.to_csv(output_file, index=False)

    return out_df