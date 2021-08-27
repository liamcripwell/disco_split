import re
import string

import nltk

from disco_split.processing.connectives import PATTERNS, INVERSES
from disco_split.models.utils import calculate_bleu, calculate_sent_bleu


def reduce_string(s):
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()

def parse_rels(sample):
    """Parses pattern tree for given example."""
    relation = PATTERNS[sample.rel]
    sense = relation[sample.sense]
    pattern = sense[sample.connective]
    
    return relation, sense, pattern

def connective_present(connective, sense, sample):
    """Checks if specified connective is found in last sentence of prediction."""
    pattern = sense[connective]
    sents = [s.strip().replace("\n", "") for s in nltk.sent_tokenize(sample.pred)]

    if sents == []:
        return False, connective

    end_sent = reduce_string(sents[-1])
    match = re.search(pattern, end_sent)
    matched = match is not None

    return matched, match if match is None else connective

def correct_sense(sample, sense):
    """Checks if prediction contains connective of correct sense."""
    sense_acc = False
    for sibling in sense:
        sense_acc, pred = connective_present(sibling, sense, sample)
        if sense_acc:
            return True, pred
    return False, None

def correct_relation(sample, relation):
    """Checks if prediction contains connective of correct relation."""
    rel_acc = False
    for sense in relation.values():
        rel_acc, pred = correct_sense(sample, sense)
        if rel_acc:
            return True, pred
    return False, None

def calculate_bleus(sample):
    """Computes and unpacks BLEU scores."""
    if "sent1" in sample:
        ground_truth = " ".join([sample.sent1, sample.sent2])
    else:
        ground_truth = sample.simple
    bleu = calculate_bleu([sample.pred], [ground_truth])
    sent_bleu = calculate_sent_bleu([sample.pred], [ground_truth])
    return {**bleu, **sent_bleu}

def is_accurate(pred_connective, bleu, sent_bleu, ref_connective):
    # NOTE: we are currently using sense as analogue for equivalency
    sense = [sense for rel in PATTERNS.values() for sense in rel.values() if ref_connective in sense]
    assert len(sense) == 1
    sense = sense[0]
    
    if bleu >= 50:
        if sent_bleu > 10:
            if pred_connective == ref_connective or pred_connective in sense:
                return True
        else:
            if ref_connective in INVERSES and pred_connective in INVERSES[ref_connective]:
                return True
    return False

def print_eval(df):
    print("BLEU-Score     |", df.bleu.mean())
    print("Sent-BLEU-Score|", df.sent_bleu.mean())
    if "samsa" in df.columns:
        print("SAMSA          |", df.samsa.mean())
    print("Relation Acc   |", df.relation_acc.mean())
    print("Connective Acc |", df.connective_acc.mean())
    print("Full Acc       |", df.bin_acc.mean())