import argparse

import numpy as np
import torch
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import datasets
import random
import os
import tqdm

from datetime import datetime



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="anli-phrase-augmented")
    parser.add_argument("-m", "--model", dest="base_model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("-r", "--report", dest="report", default=False, action="store_true")
    parser.add_argument("--lr", dest="learning_rate", default=5e-06, type=float)
    parser.add_argument("--aug", dest="augmentations", default=None, type=lambda x: x.split(","))
    parser.add_argument("-s", "--steps", default=20000, type=int, help="Number of training steps")
    parser.add_argument("--sample", default=None, type=int, help="Sample index to train on")
    parser.add_argument("--sample-phrases", default=False, action="store_true")
    parser.add_argument("--ablate-aug-type", default=None, type=lambda x: x.split(","))

    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def preprocess(entry):
        out_dict = tokenizer(entry["premise"], entry["hypothesis"], truncation=True, max_length=512)
        out_dict["label"] = entry["label"]
        return out_dict

    aug_str = "all"
    if args.augmentations is not None:
        aug_str = "+".join(args.augmentations)
        dataset = dataset.filter(lambda x: x["augment_type"] in args.augmentations)

    if args.sample is not None:
        aug_str += f"_sample-{args.sample}"
        dataset = dataset.filter(lambda x: x["sample_idx"] in (0, args.sample))
    
    print("Dataset size:", len(dataset))
    if args.sample_phrases:
        all_phrase_indices = set()
        for entry in tqdm.tqdm(dataset["train"]):
            if entry["augment_type"] == "dialog":
                all_phrase_indices.add(entry["phrase_idx"])

        sampled_phrase_indices = random.sample(list(all_phrase_indices), len(all_phrase_indices) // 2)

        aug_str += "_sampled"

        dataset = dataset.filter(lambda x: x["augment_type"] != "dialog" or x["phrase_idx"] in sampled_phrase_indices)
        print(">", len(dataset))
    print("Dataset after phrase sampling:", len(dataset))

    if args.ablate_aug_type is not None:
        aug_str += "_ablate-" + "+".join(args.ablate_aug_type)
        dataset = dataset.filter(lambda x: x["augment_type"] != "dialog" or x["phrase_type"] not in args.ablate_aug_type)

    print("Dataset after ablation:", len(dataset))

    dataset = dataset.map(preprocess, batched=False, remove_columns=list(dataset["train"].features))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    lr = args.learning_rate
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3)
    model_name = f"{aug_str}-{lr}"

    batch_size = 8
    grad_acc = 4

    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id is None:
        job_id = datetime.today().strftime('%Y%m%d_%H%M%S')
    saved_model_name = f"{model_name}_{job_id}"

    training_args = TrainingArguments(
        output_dir=f"nli_trainer_{saved_model_name}",
        num_train_epochs= args.steps / (len(dataset["train"]) / (batch_size * grad_acc * torch.cuda.device_count())),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,   
        gradient_accumulation_steps=grad_acc,
        per_device_eval_batch_size=64,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=True,
        report_to="wandb" if args.report else "none",
        save_strategy="steps",
        save_total_limit=10,
        save_steps=5000,
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=5000,
        load_best_model_at_end=True
    )

    def compute_metrics(eval_results):
        predicted = np.array(eval_results.predictions).argmax(axis=-1)
        return {"accuracy": (eval_results.label_ids == predicted).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        data_collator=collator,
    )
    out_dir = Path("./nli_models")
    out_dir.mkdir(exist_ok=True)

    trainer.train()

    trainer.save_model(out_dir / saved_model_name)


if __name__ == "__main__":
    main()
