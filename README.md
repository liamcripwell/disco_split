# Discourse-Based Sentence Splitting

Code for running discourse-based sentence splitting experiments.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Download Dataset
Datasets described in the paper are available to download from `data/dsplit_data.zip`. This contains all of `D_CCNews_C`, `D_CCNews_S`, `D_MUSS`, and `D_WikiSplit`.

## Building a Synthetic Dataset

Alternatively, you can build your own version of the data. You should use [this repo](https://github.com/liamcripwell/news-please) to mine sentences with discourse connectives or sentence pairs with an adverbial from the [Common Crawl News Corpus](https://commoncrawl.org/2016/10/news-dataset-available/). Once you have some mined data, you can use tools in this package to compile samples for training/testing models. 

To build a dataset:

```bash
python disco_split/processing/build_dataset.py mine <mined_articles_dir> <output_file> <sample_limit>
```

The `--item_type` arg can be used to specify whether you are extracting from a mine of `complex` or `simple` texts.

To transform a simple sentence dataset into (`T`, `S`) sequence pairs:

```bash
# NOTE: see the actual script for optional args
python disco_split/processing/build_dataset.py s2t <dataset_file> <output_file>
```

You can also use this library as an interface to the PDTB parser (original implementation [here](https://github.com/WING-NUS/pdtb-parser)), which can be used to generate trees for complex sentences.
```bash
# an example with the parsing configuration used in the paper
python disco_split/evaluation/pipeline.py prepare_trees <dataset_file> --out_trees=<trees_output> --out_parsed=<parser_output> --text_col=sentence --fallback=naive --batched=True --num_procs=8 --batch_size=128
```

In order to generate synthetic complex sentences from simple sentence pairs using our rule-based approach, run the following script:

```bash
python disco_split/processing/build_dataset.py s2c <input_file> <output_file>
```

If you would like to specifically reformat intermediate parser ouput into `T` sequences, you may use the following:

```bash
# NOTE: see the actual script for optional args
python disco_split/processing/build_dataset.py a2t <new_oracles_csv> <output_file>
```

## Training a Model

Code for training models is located in `disco_split/models/`.

To finetune a baseline BART model:

```bash
python disco_split/models/train_bart.py --data_file=<dataset_file>
        --train_split=0.95
        --max_epochs=5
        --gpus=1
        --learning_rate=3e-5
        --batch_size=16
        --max_source_length=64
        --max_target_length=64
        --eval_beams=4
        --eval_max_gen_length=64
        --max_samples=1e6
        --val_check_interval=0.2
```

Additional arguments can be found in `disco_split/models/bart.py`. These include options for multiple input files, loading checkpoints, specifying save directory, columns names, etc.

## Evaluation

You can evaluate end-to-end models from the terminal as follows:

```bash
disco_split/evaluation/evaluate.py evaluate <model_loc> <test_data> <output_file> --samsa=True
```

Alternatively, to evaluate pipeline models:

```bash
# using parser C2T component
python disco_split/evaluation/pipeline.py evaluate <t2s_model> <test_data> <output_dir> --parser=pdtb

# using trained neural C2T component
python disco_split/evaluation/pipeline.py evaluate <t2s_model> <test_data> <output_dir> --parser=bart --bart_model=<c2t_model>
```
