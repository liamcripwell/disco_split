from string import punctuation

import nltk
import pandas as pd

from disco_split.processing.oracle import set_rel_and_sense
from disco_split.processing.connectives import PATTERNS, FUNCTORS, INNERS
from disco_split.processing.utils import sense_of_conn, strip_adverbial, get_pattern

# NOTE: these assume args are sent to lambda in original order from a sentence pair.
# i.e. x[1] is always from the second sentence in the pair.
TRANSFORMS = {
    # arg1 preceeds arg2
    "precedence": lambda x: f"<DR> {FUNCTORS['precedence']} <ARG1> {x[0]} <ARG2> {x[1]} <EOS>",
    "succession": lambda x: f"<DR> {FUNCTORS['succession']} <ARG1> {x[1]} <ARG2> {x[0]} <EOS>",
    # arg1 and arg2 temporally overlap 
    "synchrony": lambda x: f"<DR> {FUNCTORS['synchrony']} <ARG1> {x[0]} <ARG2> {x[1]} <EOS>",
    # arg2 is interpreted as the result of the situation presented in arg1
    "result": lambda x: f"<DR> {FUNCTORS['result']} <ARG1> {x[0]} <ARG2> {x[1]} <EOS>",
    # arg1 and arg2 differ in respect to a shared property (no directionality)
    "contrast": lambda x: f"<DR> {FUNCTORS['contrast']} <ARG1> {x[0]} <ARG2> {x[1]} <EOS>",
    # arg2 provides additional information related to the situation described in arg1
    "conjunction": lambda x: f"<DR> {FUNCTORS['conjunction']} <ARG1> {x[0]} <ARG2> {x[1]} <EOS>",
    # arg1 evokes a set and arg2 describes it in further detail
    "instantiation": lambda x: f"<DR> {FUNCTORS['instantiation']} <ARG1> {x[0]} <ARG2> {x[1]} <EOS>",
    # ||arg1|| xor ||arg2|| NOTE: we are currently only accounting for the _chosen_alternative_ subtype
    "alternative": lambda x: f"<DR> {FUNCTORS['alternative']} <ARG1> {x[0]} <ARG2> {x[1]} <EOS>",
}


def format_tree(sense, arg1, arg2):
    """
    Transforms sentence pair into an annotated input sequence.
    """
    arg1 = arg1.strip(punctuation).strip()
    arg2 = arg2.strip(punctuation).strip()
    return TRANSFORMS[sense]((arg1, arg2))


def args_to_tree(samples, out_file=None, arg_cols=["arg1", "arg2"], conn_col="connective", make_ys=False, bin_mask=None, oracle=False):
    """
    Generates transformed sequence-to-sequence pairs from discourse relation and arguments
    """
    if isinstance(samples, str):
        samples = pd.read_csv(samples)
    if bin_mask is not None:
        samples = samples[samples[bin_mask] > 0]

    if oracle:
        samples = set_rel_and_sense(samples, conn_col=conn_col)

    xs = []
    tree_rels = []
    for _, sample in samples.iterrows():
        if oracle:
            # HACK: need to manually prevent arg flipping for succession relation as
            # we are already manually doing this in the oracle candidate selection
            if sample.sense == "succession":
                x = format_tree("precedence", sample[arg_cols[0]], sample[arg_cols[1]])
            else:
                # generate input and output sequences
                x = format_tree(sample.sense, sample[arg_cols[0]], sample[arg_cols[1]])
            xs.append(x)

        else:
            # skip for failed parse rows
            if pd.isna(sample.pred_rel):
                xs.append("")
                tree_rels.append("")
                continue

            # extract predicted rel and best approximate predicted sense
            pred_rel, pred_sense = (sample.pred_rel.split(".") + [None, None])[:2]
            possible_senses = list(PATTERNS[pred_rel].keys())

            if pred_sense is None:
                pred_sense = sense_of_conn(sample.marker)
            temp_sense = pred_sense if pred_sense in possible_senses else possible_senses[0]

            # handle edge cases
            if sample.pred_rel == "temporal.synchrony":
                temp_sense = "synchrony"
            elif sample.pred_rel == "temporal.asynchronous":
                # if no sense labels then check if known succession inner connective
                known_markers = [x for a in PATTERNS["temporal"]["succession"] for x in INNERS[a]]
                if sample.marker in known_markers:
                    temp_sense = "succession"
                else:
                    temp_sense = "precedence"

            # make trees
            x = format_tree(temp_sense, sample[arg_cols[0]], sample[arg_cols[1]])
            xs.append(x)

            # store what was used as the rel for the tree creation
            tree_rels.append(f"{pred_rel}.{temp_sense}")

    samples["tree"] = xs
    if not oracle:
        samples["tree_rel"] = tree_rels

    if out_file:
        samples.to_csv(out_file, index=False)

    return samples


def simples_to_tree(samples, out_file=None, keep_original=False, conn_col="connective"):
    """
    Generates transformed sequence-to-sequence pairs from simple sentence pair
    """

    # read samples from file
    if isinstance(samples, str):
        samples = pd.read_csv(samples)

    # transform samples
    transformed = []
    for i, sample in samples.iterrows():
        # find pattern for adverbial
        connective_pattern = get_pattern(sample.sense, sample[conn_col])

        # identify sentences and assign to args
        if "sent1" in samples.columns:
            sent1 = sample.sent1
            sent2 = sample.sent2
        else:
            sent1, sent2 = nltk.sent_tokenize(sample.simple)[:2]
        arg1 = sent1
        arg2 = strip_adverbial(connective_pattern, sent2)

        # generate input and output sequences
        input_seq = format_tree(sample.sense, arg1, arg2)
        transformed.append((input_seq, f"{sent1} {sent2}"))

    # form resulting data structure
    if keep_original:
        samples["x"] = [t[0] for t in transformed]
        samples["y"] = [t[1] for t in transformed]
        transformed = samples
    else:
        transformed = pd.DataFrame(transformed, columns=["x", "y"])

    # output transformed samples to file
    if out_file:
        transformed.to_csv(out_file, index=False)

    return transformed