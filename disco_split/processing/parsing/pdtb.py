import os
import re
import shutil
import subprocess
from multiprocessing import Pool

import fire
import numpy as np
import pandas as pd

from disco_split.processing.connectives import INNERS, FORWARDS, PDTB_RELS
from disco_split.processing.parsing.naive import parse as naive_parse

DEF_TEMP_DIR = "resources/pdtb-parser/temp"

PIPE_COLS = {
    0: "type", # Explicit, Implicit, etc.
    11: "pred_rel", # Temporal, Expension, etc.
    5: "marker",
    24: "arg1",
    34: "arg2",
}

CONN_SPAN_COL = 3
ARG_SPAN_COLS = [22, 32]


def strip_punct_conj(in_str):
    out_str = in_str.strip(" ,.;")
    out_str = re.sub(' (and|but)$', '', out_str)
    out_str = re.sub('^(and|but) ', '', out_str)
    out_str = out_str.strip(" ,.;")

    return out_str

def configure_out_dir(rerun=False, cont=False):
    """Set up the directory to house temporary data for this parsing job."""
    temp_dir = DEF_TEMP_DIR
    if rerun:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
    elif not cont:
        # allow for distinguishing simultaneous jobs
        k = 2
        while os.path.exists(temp_dir):
            temp_dir = DEF_TEMP_DIR + "_" + str(k)
            k += 1
        os.mkdir(temp_dir)
    
    return temp_dir

def extract_results(pipe_file, full_text, naive_args=False):
    """Extract parser results from _.pipe_ files."""

    sample_id = pipe_file.split("/")[-1].split(".")[0]
    results = []
    with open(pipe_file, "r") as pipes:
        for pipe in pipes.readlines():
            cols = pipe.split("|")
            if cols[0] != "Explicit":
                return None

            # lowercase the relation type and marker
            cols[11] = cols[11].lower()
            cols[5] = cols[5].lower()

            # perform naive allocation of relation arguments (arg1 is all before conn, arg2 is all after)
            if (naive_args and len(full_text) > 1) or any([arg is None for arg in [cols[24], cols[34]]]):
                if any([len(arg.split()) < 3 for arg in [cols[24], cols[34]]]) and ";" not in cols[CONN_SPAN_COL]:
                    span = [int(x) for x in cols[CONN_SPAN_COL].split("..")]
                    cols[24] = full_text[:span[0]]
                    cols[34] = full_text[span[1]:]
            
            # strip trailing punctuation and conjunctions from args
            cols[24] = strip_punct_conj(cols[24])
            cols[34] = strip_punct_conj(cols[34])

            results.append([sample_id] + [cols[k] for k, v in PIPE_COLS.items()])

    return results

def filtered_results(example, results, strict=True, text_col="complex"):
    CONNS = { k: INNERS[k]+FORWARDS[k] for k in INNERS.keys() }

    to_keep = []
    if "rel" in example.axes[0]:
        for result in results:
            # NOTE: index 2 here refers to the parsed/predicted relation
            if result[2] in PDTB_RELS[example.rel]:
                to_keep.append(result)
    elif "adverbial" in example.axes[0]:
        # enforce results that are valid intrasent connective for known adverbial
        for result in results:
            # NOTE: index 3 here refers to the parsed/predicted marker
            if result[3] == example.adverbial or result[3] in INNERS[example.adverbial]:
                to_keep.append(result)
    elif "connective" in example.axes[0]:
        # enforce only results with specific known connective
        for result in results:
            if result[3] == example.connective:
                to_keep.append(result)
    elif strict:
        # exclude those that have an unknown/unsupported marker
        markers = np.concatenate(list(CONNS.values()))
        for result in to_keep:
            if result[3] in markers:
                to_keep.append(result)
    else:
        print("No relation labels, considering all parsed results...")
        to_keep = results

    if len(to_keep) > 1:
        # keep one closest to covering full sentence with smallest difference in length between its arguments
        to_keep = [sorted(to_keep, key=lambda x: 
                        abs(len(x[4]) - len(x[5])) + 
                        abs(len(x[4] + x[5]) - len(example[text_col]))
                        )[0]]

    return to_keep

def parsing_pool(in_file, batch_size=1e4, num_procs=1, args=()):
    """Run _parse_samples()_ in multiproc batches."""
    if isinstance(in_file, str):
        df_in = pd.read_csv(in_file)
    else:
        df_in = in_file

    num_splits = int(len(df_in) / batch_size)
    dfs = np.array_split(df_in, num_splits + 1)
    dfs = [df.reset_index() for df in dfs]
    out_file_base = args[1].split(".")[0]
    temp_dirs = [configure_out_dir() for df in dfs]
    args_list = [(df, f"{out_file_base}_{i}.csv") + args[2:] + (temp_dirs[i],) for i, df in enumerate(dfs)]

    with Pool(num_procs) as pool:
        results = pool.starmap(parse_samples, args_list)

    res_df = pd.concat(results)
    res_df.to_csv(args[1], index=False)

    return res_df

def parse_samples(in_file, out_file=None, text_col="text", keep_cols=[], filter_results=False, naive_args=False, 
                    fallback=None, rerun=False, cont=False, temp_dir=None):
    """Run batch of samples through the PDTB parser."""

    if isinstance(in_file, str):
        df_in = pd.read_csv(in_file)
    else:
        df_in = in_file

    if temp_dir is None:
        temp_dir = configure_out_dir(rerun, cont)

    if not cont:
        # prepare a file for each sample that will be read by the parser
        for i, sample in df_in.iterrows():
            with open(f"{temp_dir}/{i}.txt", "w") as text_file:
                text_file.write(sample[text_col])

    # run parser on each sample
    os.chdir("resources/pdtb-parser/")
    subprocess.run(["java", "-jar", "parser.jar", temp_dir.split("/")[-1]])
    os.chdir("../../")

    # extract results from each sample's output file
    parsed_results = []
    for i, row in df_in.iterrows():
        results = extract_results(f"{temp_dir}/output/{i}.txt.pipe", str(row[text_col]), naive_args=naive_args)
        for result in results:
            result.append(row[text_col])

        # filter bad parser results
        if filter_results:
            results = filtered_results(row, results, text_col=text_col)

        if results == []:
            # fallback to naive parser if no good results`
            if fallback == "naive":
                pred_rel, marker, args = naive_parse(row[text_col])
                results.append([i] + [None, pred_rel, marker] + args + [df_in.iloc[i][text_col]])
            else:
                results.append([i] + [None for _ in PIPE_COLS] + [df_in.iloc[i][text_col]])

        parsed_results += results

    # prepare output data frame
    df_out = pd.DataFrame(parsed_results, columns=["sample"] + [v for k, v in PIPE_COLS.items()] + [text_col])
    for col in keep_cols:
        if col not in df_in.columns: continue
        ids = list(df_out["sample"].astype(int))
        new_col = list(df_in.iloc[ids][col])
        df_out.insert(loc=0, column=col, value=new_col)

    if out_file is not None:
        df_out.to_csv(out_file, index=False)

    # delete parser temporary data
    shutil.rmtree(temp_dir)

    return df_out


if __name__ == '__main__':
    fire.Fire(parse_samples)
