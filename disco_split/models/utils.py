import itertools
from typing import Callable, Iterable, List

import nltk
from torch import nn
from sacrebleu import corpus_bleu


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart."""
    freeze_params(model.model.shared)
    for d in [model.model.encoder, model.model.decoder]:
        freeze_params(d.embed_positions)
        freeze_params(d.embed_tokens)


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {
        "bleu": round(
            corpus_bleu(
                output_lns,
                [refs_lns],
                **kwargs).score,
            4)}

def calculate_sent_bleu(output_lns, refs_lns, **kwargs) -> dict:
    assert(len(output_lns) == len(refs_lns))

    out_firsts = []
    ref_firsts = []
    out_seconds = []
    ref_seconds = []
    for i in range(len(output_lns)):
        out_sents = nltk.sent_tokenize(output_lns[i])
        ref_sents = nltk.sent_tokenize(refs_lns[i])

        # artificially pad to 2 sentences for each
        out_sents += [""] * (2 - len(out_sents))
        ref_sents += [""] * (2 - len(ref_sents))

        if len(out_sents) < 2: print(out_sents)
        if len(ref_sents) < 2: print(ref_sents)

        out_firsts.append(out_sents[0])
        out_seconds.append(out_sents[1])
        ref_firsts.append(ref_sents[0])
        ref_seconds.append(ref_sents[1])

    sent1_bleu = round(
        corpus_bleu(
            out_firsts,
            [ref_firsts],
            **kwargs).score,
        4)
    sent2_bleu = round(
        corpus_bleu(
            out_seconds,
            [ref_seconds],
            **kwargs).score,
        4)

    return {
        "sent1_bleu": sent1_bleu,
        "sent2_bleu": sent2_bleu,
    }
