import fire
from transformers import BartTokenizer, BartForConditionalGeneration

from disco_split.models.bart import BartFinetuner
from disco_split.evaluation.utils import print_eval
from disco_split.evaluation import run_predictions, run_evaluation


def load_model(model_loc, base=None, base_tokenizer=False, new_tokens="mt"):
    if base_tokenizer:
        # load pretained model and tokenizer
        tokenizer = BartTokenizer.from_pretrained(
            base, add_prefix_space=True)
        bart_model = BartForConditionalGeneration.from_pretrained(
            base, return_dict=True)

        model = BartFinetuner.load_from_checkpoint(
            model_loc, tokenizer = tokenizer, base_model = bart_model)
    else:
        model = BartFinetuner.load_from_checkpoint(model_loc, new_tokens=new_tokens)

    model.eval()
    return model

def evaluate(model_loc, test_file, output_file, input_col="complex", label_col="simple", 
                task=None, batch_size=16, samsa=False):
    model = load_model(model_loc, 'facebook/bart-base')

    df = run_predictions(model, test_file, input_col=input_col, label_col=label_col, 
                            task=task, batch_size=batch_size)
    df = run_evaluation(df, output_file=output_file, samsa=samsa)

    print_eval(df)


if __name__ == '__main__':
    fire.Fire()