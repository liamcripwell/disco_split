import re
import random

import spacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")

from disco_split.processing.utils import strip_adverbial, get_pattern
from disco_split.processing.connectives import INNERS, FORWARDS


def build_complex_inner(arg1, arg2, conn):
    # chance to put comma between first arg and connective
    roll = random.uniform(0, 1)
    if roll < 0.25:
        arg1 += ","

    # remove start of sentence capital from arg2
    doc = nlp(arg2)
    if doc[0].pos_ != "PROPN":
        arg2 = arg2[0].lower() + arg2[1:]
    
    return f"{arg1} {conn} {arg2}"

def build_complex_forward(arg1, arg2, conn):
    reverse = re.search("^<REV>", conn) is not None
    conn = re.sub("<REV>", "", conn)
    conn = conn[0].upper() + conn[1:]
    
    # handle case resolution
    doc = nlp(arg1)
    if doc[0].pos_ != "PROPN":
        arg1 = arg1[0].lower() + arg1[1:]
    doc = nlp(arg2)
    if doc[0].pos_ != "PROPN":
        arg2 = arg2[0].lower() + arg2[1:]
    
    # arrange sentence
    if reverse:
        result = f"{conn} {arg1}, {arg2}"
    else:
        result = f"{conn} {arg2}, {arg1}"
    
    return result

def simples_to_complex(samples, out_file=None, complex_col="complex", simp_cols=["sent1", "sent2"], seed=False):
    """
    Generates complex sentences from simple sentence pairs
    """
    if isinstance(samples, str):
        samples = pd.read_csv(samples)

    if seed is not None:
        random.seed(seed)

    complexes = []
    for _, sample in samples.iterrows():
        connective_pattern = get_pattern(sample.sense, sample.connective)
        if connective_pattern is None:
            complexes.append("")
            continue
        
        if FORWARDS[sample.connective] != []:
            # randomly decide between an inner or forward connective structure if both available
            options = random.choice([INNERS, FORWARDS])
        else:
            options = INNERS
        complex_conn = random.choice(options[sample.connective])

        # construct complex sentence
        arg1 = sample[simp_cols[0]].strip(".?!")
        arg2 = strip_adverbial(connective_pattern, sample[simp_cols[1]])
        if arg1 == "" or arg2 == "":
            complexes.append(None)
            continue
        if options is INNERS:
            complex_sent = build_complex_inner(arg1, arg2, complex_conn)
        else:
            complex_sent = build_complex_forward(arg1, arg2, complex_conn)

        complexes.append(complex_sent)

    samples[complex_col] = complexes

    if out_file:
        samples.to_csv(open(out_file, "w"), index=False)