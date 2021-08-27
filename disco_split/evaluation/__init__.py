import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from disco_split.evaluation.utils import *
from disco_split.models.bart import BartDataModule


def run_predictions(model, input_file, output_file=None, input_col="x", label_col="y", task=None, batch_size=16):
    """
    Perform prediction with model on given test set.
    """
    if isinstance(input_file, str):
        test_set = pd.read_csv(input_file)
    else:
        test_set = input_file

    # add task functor to start of input sequences when using multi-task model
    if task is not None:
        test_set[input_col] = test_set[input_col].replace("^", f"{task} ", regex=True)

    dm = BartDataModule(model.tokenizer, hparams=model.hparams)
    test = dm.transform(list(test_set[input_col]), list(test_set[label_col]))
    dataset = TensorDataset(
            test['input_ids'],
            test['attention_mask'],
            test['labels'])
    test_data = DataLoader(dataset, batch_size=batch_size)

    pred_ys = []
    for batch in test_data:
        results = model._generative_step(batch)
        pred_ys += results["preds"]

    test_set["pred"] = pred_ys

    if output_file is not None:
        test_set.to_csv(output_file, index=False)

    return test_set


def run_evaluation(test_set, output_file=None, samsa=False):
    conn_accs = []
    rel_accs = []
    bleus = []
    sent_bleus = []
    bin_accs = []
    pred_conns = []
    samsas = []

    # we don't calculate SAMSA by default as it is very long to compute
    if samsa:
        from easse.samsa import get_samsa_sentence_scores
        samsas = get_samsa_sentence_scores(list(test_set.complex), list(test_set.pred))

    for _, sample in test_set.iterrows():
        relation, sense, _ = parse_rels(sample)

        matched, pred_conn = connective_present(sample.connective, sense, sample)
        s_matched, pred_conn = (matched, pred_conn) if matched else correct_sense(sample, sense)
        pred_conns.append(pred_conn)

        bleu_scores = calculate_bleus(sample)
        bleu = bleu_scores["bleu"]
        sent_bleu = (bleu_scores["sent1_bleu"] + bleu_scores["sent2_bleu"]) / 2
        
        bin_acc = is_accurate(pred_conn, bleu, sent_bleu, sample.connective)

        conn_accs.append(matched)
        # NOTE: what is referred to as `R_Acc` is actually based on the `sense` in our code
        rel_accs.append(s_matched)
        bleus.append(bleu)
        sent_bleus.append(sent_bleu)
        bin_accs.append(bin_acc)

    test_set["connective_acc"] = conn_accs
    test_set["relation_acc"] = rel_accs
    test_set["bleu"] = bleus
    test_set["sent_bleu"] = sent_bleus
    test_set["bin_acc"] = bin_accs
    test_set["pred_connective"] = pred_conns
    if samsa:
        test_set["samsa"] = samsas

    if output_file is not None:
        test_set.to_csv(output_file, index=False)
    
    return test_set