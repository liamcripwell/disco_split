import os

import fire
import pandas as pd

from disco_split.evaluation.utils import print_eval
from disco_split.evaluation.evaluate import load_model
from disco_split.evaluation import run_predictions, run_evaluation
from disco_split.processing.transforms.tree import args_to_tree
from disco_split.processing.parsing.pdtb import parsing_pool as pdtb_pool
from disco_split.processing.parsing.pdtb import parse_samples as pdtb_parse
from disco_split.processing.parsing.naive import parse_samples as naive_parse
from disco_split.processing.parsing.bart import BartParser


def prepare_trees(samples, parser="pdtb", out_parsed=None, out_trees=None, text_col="text", n=None, naive_args=False, 
                    fallback=None, rerun=False, cont=False, bart_model=None, batched=False, batch_size=1e4, num_procs=1):
    if isinstance(samples, str):
        samples = pd.read_csv(samples)
    
    # limit number of examples if necessary
    if n is not None:
        samples = samples.iloc[:n]

    print("Parsing discourse structure...")
    if parser == "pdtb":
        args = (
            samples,
            out_parsed,
            text_col,
            ["connective", "sense", "rel", "simple", "source_id", "source"],
            True,
            naive_args,
            fallback,
            rerun,
            cont,
        )
        if batched:
            df_parsed = pdtb_pool(samples, batch_size, num_procs, args)
        else:
            df_parsed = pdtb_parse(*args)
    elif parser == "naive":
        df_parsed = naive_parse(samples, out_parsed, text_col=text_col)
    elif parser == "bart":
        # NOTE: the Bart parser creates trees in a single step
        bart_parser = BartParser(bart_model)
        df_trees = bart_parser.parse_samples(samples, text_col=text_col)
    else:
        raise ValueError(f"Could not interpret specified parser \"{parser}\"...")

    if parser != "bart":
        # transform parsed units into linearized tree
        print("Transforming parsed data into tree...")
        df_trees = args_to_tree(df_parsed, out_file=out_trees)

    return df_trees

def evaluate(model_loc, test_file, out_dir, parser="pdtb", text_col="text", n=None, bart_model=None, naive_args=False, fallback=None, overwrite=False):
    # create output directory
    out_dir = out_dir.strip("/")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if isinstance(test_file, str):
        samples = pd.read_csv(test_file)
    else:
        samples = test_file

    # limit number of examples if necessary
    if n is not None:
        samples = samples.iloc[:n]

    # run discourse parser
    if overwrite or not os.path.exists(f"{out_dir}/trees.csv"):
        df_trees = prepare_trees(samples, parser=parser, out_parsed=f"{out_dir}/parsed.csv", out_trees=f"{out_dir}/trees.csv", 
                                    bart_model=bart_model, text_col=text_col, naive_args=naive_args, fallback=fallback)
    else:
        df_trees = pd.read_csv(f"{out_dir}/trees.csv")

    # load simplification model
    # NOTE: we must manually set _new_tokens_ value if we are using the old checkpoint of tree-bart
    model = load_model(model_loc, new_tokens="tree")

    # generate predictions
    print("Running sentence pair generation...")
    df_preds = run_predictions(model, df_trees, output_file=f"{out_dir}/preds.csv", input_col="tree", label_col="simple")

    # evaluate
    print("Evaluating performance...")
    df_res = run_evaluation(df_preds, output_file=f"{out_dir}/eval.csv")

    print_eval(df_res)

    return df_res


if __name__ == '__main__':
    fire.Fire()
