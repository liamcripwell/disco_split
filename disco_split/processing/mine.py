import os
import re
import json

import pandas as pd

from disco_split.processing.connectives import STRICT_PATTERNS


def write_to_file(samples, out_file, item_type="simple"):
    if out_file is None: return

    if item_type == "simple":
        cols = ["connective", "sense", "sent1", "sent2"]
    else:
        cols = ["connective", "adverbial", "type", "sentence"]

    # construct dataframe of samples
    df = pd.DataFrame(
        samples,
        columns=cols)

    # output to csv file
    df.to_csv(open(out_file, "w"), index=False)

def extract_simple(item, strict=False):
    """Extract content from an S from mine."""
    # if applying strict patterns, ignore item if no match
    if strict and item["connective"] in STRICT_PATTERNS.keys():
        if re.search(STRICT_PATTERNS[item["connective"]], item["sentence2"].lower()) is None:
            return None

    return (item["connective"], item["sense"],
                item["sentence1"], item["sentence2"])

def extract_complex(item):
    """Extract content from a C from mine."""
    if any([x not in item for x in ["connective", "adverbial", "type", "sentence"]]):
        return None

    return (item["connective"], item["adverbial"],
                item["type"], item["sentence"])

def from_mine(mine_dir, out_file=None, item_type="simple", sample_limit=None, strict=False):
    """
    Build a dataset from mined CC-News samples.
    """
    done = False
    samples = []
    if strict:
        print("Using strict pattern filtering.")

    # look through all mined articles
    for root, dirs, files in os.walk(mine_dir):
        if done: break
        for name in files:
            if done: break
            if name.endswith(".json"):
                filepath = os.path.join(root, name)
                items = json.load(open(filepath, "r"))

                # record sample
                for item in items:
                    if sample_limit is not None and len(samples) >= sample_limit:
                        done = True
                        break
                    else:
                        if item_type == "simple":
                            result = extract_simple(item, strict=strict)
                        elif item_type == "complex":
                            result = extract_complex(item)
                        else:
                            raise ValueError("Specified an unknown item type!")

                        if result is None: 
                            continue

                        samples.append(result)

                        if len(samples) % 1000 == 0:
                            print(f"Dataset at {len(samples)} samples...")

                        if len(samples) % 1e4 == 0:
                            write_to_file(samples, out_file, item_type=item_type)

    write_to_file(samples, out_file, item_type=item_type)