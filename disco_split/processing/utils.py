import re
from string import punctuation

from disco_split.processing.connectives import PATTERNS, INNERS, FORWARDS


def strip_adverbial(adverbial, sentence):
    """
    Removes the adverbial connective part of a sentence.
    """
    return re.sub(adverbial, "", sentence,
        flags=re.IGNORECASE).lstrip(punctuation).strip()

def find_adverbial(sentence, full_output=False, loose=False):
    for rel, senses in PATTERNS.items():
        for sense, conns in senses.items():
            for conn, pattern in conns.items():
                if loose:
                    pattern = pattern[1:]
                if re.search(pattern, sentence.lower()) is not None:
                    if full_output:
                        return conn, sense, rel
                    else:
                        return conn
    if full_output:
        return None, None, None
    else:
        return None

def find_inner_connective(sentence, multi=False):
    matched = []
    for adverbial, inners in INNERS.items():
        for inner in list(set([adverbial] + inners)):
            pattern = f"(,? {inner},? )"
            match = re.search(pattern, sentence)
            if match is not None:
                if multi:
                    matched.append(inner)
                else:
                    return inner
    return matched if multi else None

def find_forward_connective(sentence):
    for adverbial, conns in FORWARDS.items():
        for conn in conns:
            pattern = f"(^{conn} (?!,))"
            match = re.search(pattern, sentence.lower())
            if match is not None:
                return conn
    return None

def sense_of_conn(connective):
    """Lookup the relation sense for a given non-adverbial connective."""
    for adverbial, inners in INNERS.items():
        if connective in inners:
            for rel, senses in PATTERNS.items():
                for sense, conns in senses.items():
                    if adverbial in conns:
                        return sense
    return None

def get_pattern(sense, conn):
    """Returns the RegEx pattern for a given relation sense and connective."""
    connective_pattern = None
    for _, abstract_sense in PATTERNS.items():
        if sense in abstract_sense:
            if conn in abstract_sense[sense]:
                connective_pattern = abstract_sense[sense][conn]
                
    return connective_pattern
