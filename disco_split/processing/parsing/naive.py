import re

import fire
import pandas as pd

from disco_split.processing.connectives import PATTERNS, INNERS

# prepare reverse index of rels from conns for convenience
RELS = {}
for rel, senses in PATTERNS.items():
    for sense, conns in senses.items():
        for conn in conns.keys():
            RELS[conn] = f"{rel}.{sense}"

def strip_punct_conj(in_str):
    out_str = in_str.strip(" ,.;")
    out_str = re.sub(' and$', '', out_str)
    out_str = re.sub('^and ', '', out_str)
    out_str = out_str.strip(" ,.;")

    return out_str

def parse(text):
    marker = None
    pred_rel = None
    args = [None, None]

    for adverbial, inners in INNERS.items():
        for inner in [adverbial] + inners:
            pattern = f"(,? {inner},? )"
            match = re.search(pattern, text)

            if match is not None:
                marker = inner
                pred_rel = RELS[adverbial]

                matches = []
                for match in re.finditer(pattern, text):
                    matches.append(match.span())
                
                # choose match closest to center of text
                match_span = sorted(matches, key=lambda x: abs(x[0]+ ((x[1]-x[0])/2)) - (len(text)/2) )[0]
                args = [text[:match_span[0]], text[match_span[1]:]]
                args = [strip_punct_conj(a) for a in args]
                assert len(args) == 2

                if adverbial in PATTERNS["temporal"]["succession"]:
                    args = [args[1], args[0]]

                return pred_rel, marker, args

    return pred_rel, marker, args


def parse_samples(in_file, out_file=None, text_col="text"):
    """Run batch of samples through the naive discourse parser."""

    if isinstance(in_file, str):
        df_in = pd.read_csv(in_file)
    else:
        df_in = in_file

    # run parser on each sample
    markers = []
    pred_rels = []
    args = []
    for _, row in df_in.iterrows():
        pred_rel, marker, _args = parse(row[text_col])
        markers.append(marker)
        pred_rels.append(pred_rel)
        args.append(_args)

    # prepare output data frame
    df_out = df_in
    df_out["pred_rel"] = pred_rels
    df_out["marker"] = markers
    df_out["arg1"] = [a[0] for a in args]
    df_out["arg2"] = [a[1] for a in args]

    if out_file is not None:
        df_out.to_csv(out_file, index=False)

    return df_out


if __name__ == '__main__':
    fire.Fire(parse_samples)
