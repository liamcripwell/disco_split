import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from disco_split.models.bart import BartDataModule
from disco_split.evaluation.evaluate import load_model


class BartParser():

    def __init__(self, saved_model, base_model="facebook/bart-base"):
        self.model = load_model(saved_model, base_model)
        self.dm = BartDataModule(self.model.tokenizer, hparams=self.model.hparams)

    def parse_samples(self, df_in, text_col="complex"):
        trans = self.dm.transform(list(df_in[text_col]), list(df_in[text_col]))
        dataset = TensorDataset(
                trans['input_ids'],
                trans['attention_mask'],
                trans['labels'])
        test_data = DataLoader(dataset, batch_size=16)

        preds = []
        for batch in test_data:
            results = self.model._generative_step(batch)
            preds += results["preds"]
        df_in["tree"] = preds

        return df_in