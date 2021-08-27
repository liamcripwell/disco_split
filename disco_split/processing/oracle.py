

from disco_split.processing.connectives import PATTERNS

def set_rel_and_sense(samples, conn_col="connective"):
    # NOTE: this is revising old-style data where we used "class" instead of "rel"
    if "rel" not in samples.columns and "class" in samples.columns:
        rels = []
        for i, sample in samples.iterrows():
            rels.append(sample["class"].lower())
        samples["rel"] = rels

    # discover relation sense (assuming this isn't in data frame)
    if "sense" not in samples.columns and "rel" in samples.columns:
        senses = []
        samp_sense = None
        for _, sample in samples.iterrows():
            for sense, s in PATTERNS[sample.rel].items():
                if sample[conn_col] in s:
                    samp_sense = sense
                    break
            if samp_sense is not None:
                senses.append(sense)
            else:
                raise ValueError(f"The relation \"{sample.rel}\" does not contain connective \"{sample[conn_col]}\"")
        samples["sense"] = senses

    return samples