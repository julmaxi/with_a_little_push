from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import argparse
import tqdm
import json
import datasets
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import math
from pathlib import Path



def get_predictions(model, tokenizer, dataset, device):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, 2, shuffle=False, collate_fn=collator)
    out = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            preds = torch.nn.functional.softmax(logits, dim=-1).cpu().tolist()

            out.extend([p for p in preds])

    return np.array(out)

def get_predictions_multisample(model, tokenizer, dataset, device, n_samples=10):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, 1, shuffle=False, collate_fn=collator)
    out = []
    with torch.no_grad():
        batch_size = 2
        for batch in tqdm.tqdm(loader):
            local_out = []
            for _ in range(int(math.ceil(n_samples / batch_size))):
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                preds = torch.nn.functional.softmax(logits, dim=-1).cpu().tolist()
                local_out.append(preds)
            
            out.extend([p for p in np.mean(local_out, axis=0)])

    return np.array(out)


def make_dropout_func(do_rate):
    def f(x):
        mask = (torch.rand_like(x) < do_rate)
        return x.masked_fill(mask, 0.0) * (1.0 / (1 - do_rate))
    return f

def enable_dropout(module):
    for c in module.children():
        if c.__class__.__name__ == "StableDropout":
            if c.drop_prob > 0:
                c.forward = make_dropout_func(c.drop_prob)
        enable_dropout(c)

def compute_roc_aucs(predictions, labels, corpora):
    instances_by_corpus = defaultdict(lambda: ([], []))
    for pred, label, corpus in zip(predictions, labels, corpora):
        preds, labels = instances_by_corpus[corpus]
        preds.append(pred)
        labels.append(label)

    out = {}
    
    for corpus, (preds, labels) in instances_by_corpus.items():
        out[corpus] = roc_auc_score(labels, preds)

    avg = sum(out.values()) / len(out)

    return avg, out

@dataclass
class PreparedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification

    @staticmethod
    def add_default_args(parser):
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--tokenizer", type=str, default="microsoft/deberta-v3-large")

    @classmethod
    def from_args(cls, args):
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model.eval()
        model.to("cuda")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        return cls(tokenizer=tokenizer, model=model)


def prepare_dataset(dataset, out_columns, premise_key, hypothesis_key, tokenizer):
    def preprocess(entry):
        out_dict = tokenizer(entry[premise_key], entry[hypothesis_key], truncation="only_first", max_length=512)
        return out_dict
    dataset = dataset.map(preprocess, batched=False)

    out_entries = {k: [] for k in out_columns}
    for entry in dataset:
        for k in out_columns:
            out_entries[k].append(entry[k])
    
    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in dataset[0]:
        columns.append("token_type_ids")
    dataset.set_format(type="torch", columns=columns)
    return dataset, out_entries


@dataclass
class PreparedDataset:
    dataset: datasets.Dataset = None
    prepared_model: PreparedModel = None
    corpora: list[str] = None
    labels: list[int] = None

    @staticmethod
    def add_default_args(parser):
        PreparedModel.add_default_args(parser)
        parser.add_argument("--dataset", type=Path, default="true.json")

    @classmethod
    def from_args(cls, args):
        prepared_model = PreparedModel.from_args(args)

        tokenizer = prepared_model.tokenizer

        dataset = datasets.load_dataset('json', data_files=str(args.dataset))
        dataset = dataset["train"]

        dataset, out_columns = prepare_dataset(dataset, ["label", "corpus"], "grounding", "generation", tokenizer)

        labels = out_columns["label"]
        corpora = out_columns["corpus"]
        
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        return cls(dataset=dataset, labels=labels, corpora=corpora, prepared_model=prepared_model)
    
    @property
    def tokenizer(self):
        return self.prepared_model.tokenizer
    
    @property
    def model(self):
        return self.prepared_model.model


def main():
    parser = argparse.ArgumentParser()
    PreparedDataset.add_default_args(parser)
    parser.add_argument("-m", "--mode", choices=["e", "e-c"], default="e-c")
    parser.add_argument("--mc", dest="enable_dropout", action="store_true", default=False)
    args = parser.parse_args()
    dataset = PreparedDataset.from_args(args)

    if args.enable_dropout:
        predictions = get_predictions_multisample(dataset.model, dataset.tokenizer, dataset.dataset, "cuda")
    else:
        predictions = get_predictions(dataset.model, dataset.tokenizer, dataset.dataset, "cuda")

    if args.mode == "e-c":
        scores = predictions[:,0] - predictions[:,2]
    else:
        scores = predictions[:,0]

    dataset_name = args.dataset.name.rsplit(".", 1)[0]

    out_dir = Path(f"predicted_scores_{dataset_name}")
    out_dir.mkdir(exist_ok=True)

    model_basename = args.model.rstrip("/").rsplit("/", 1)[-1]
    if args.enable_dropout:
        model_basename += "_mc"
    with open(out_dir / model_basename, "w") as f:
        pd.DataFrame({"prediction": predictions.tolist(), "scores": scores, "label": dataset.labels, "corpus": dataset.corpora}).to_csv(f, index=False)

    avg, out = compute_roc_aucs(scores, dataset.labels, dataset.corpora)

    for key, score in out.items():
        print(f"{key} {score:.2f}")

    print(f"{avg:.2f}")


if __name__ == "__main__":
    main()
