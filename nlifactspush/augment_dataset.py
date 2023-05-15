import datasets
import random
from collections import defaultdict

from typing import Callable, Any
from transformers import BartForConditionalGeneration, AutoTokenizer
import torch


def batched_instance_fn(f: Callable[
    [dict[str, Any]],
    list[dict[str, Any]]
]):
    def inner(instances: dict[str, list], **kwargs) -> dict[str, list]:
        keys = list(instances.keys())
        new_keys = ["sample_idx", "augment_type", "phrase_idx", "phrase_type"]
        new_keys.extend(instances.keys())
        n_instances = len(instances[keys[0]])
        out_dict = {k: [] for k in new_keys}
        for idx in range(n_instances):
            d = {k: instances[k][idx] for k in keys}
            out_dicts = f(d, **kwargs)
            if len(out_dicts) == 0:
                continue
            for out in out_dicts:
                for k in new_keys:
                    out_dict[k].append(out[k])
        print([(k, len(v)) for k, v in out_dict.items()])
        return out_dict

    return inner


PHRASES = list(enumerate([
    ("I am not sure, but ", "hedging"),
    ("I am not sure but I do know that ", "hedging"),
    ("I believe ",  "hedging"),
    ("I do not have information on this but ", "hedging"),
    ("I think ", "hedging"),
    ("Here is what I know: ", "introduction"),
    ("yep. Also ", "introduction"),
    ("Sure! Here is what I know: ", "introduction"),
    ("I love that! ", "opinion"),
    ("I like that! ", "opinion")
]))


def augment_dataset(dataset: datasets.arrow_dataset.Dataset, phrases=PHRASES, ratio=1):
    
    @batched_instance_fn
    def phrase_augment_fn(instance: dict[str, Any], n_samples=1) -> dict[str, Any]:
        hypothesis = instance["hypothesis"]
        confounders = random.sample(phrases, k=n_samples)

        result = []

        for idx, (phrase_idx, (phrase, phrase_type)) in enumerate(confounders):
            out = dict(instance)
            out["augment_type"] = "dialog"
            out["phrase_idx"] = phrase_idx
            out["phrase_type"] = phrase_type
            out["sample_idx"] = 1 + idx
            sample = phrase + hypothesis[0].lower() + hypothesis[1:]
            out["hypothesis"] = sample
            result.append(out)
        
        return result

    @batched_instance_fn
    def original_fn(instance: dict[str, Any], **kwargs):
        out = dict(instance)
        out["augment_type"] = "original"
        out["sample_idx"] = 0
        out["phrase_idx"] = 0
        out["phrase_type"] = ""
    
        return [out]

    augment_functions = [original_fn]
    augment_functions.append(
        phrase_augment_fn
    )

    return datasets.concatenate_datasets([
        dataset.map(
            lambda x: f(x, n_samples=1),
            batched=True,
            load_from_cache_file=False,
        ) for f in augment_functions
    ])


def main():
    dataset = datasets.load_dataset("anli")
    dataset = {key: augment_dataset(d) for key, d in dataset.items()}

    dataset = datasets.dataset_dict.DatasetDict(
        {
            "train": datasets.concatenate_datasets(
                [dataset["train_r1"], dataset["train_r2"], dataset["train_r3"]]
            ),
            "dev": datasets.concatenate_datasets(
                [dataset["dev_r1"], dataset["dev_r2"], dataset["dev_r3"]]
            ),
            "test": datasets.concatenate_datasets(
                [dataset["test_r1"], dataset["test_r2"], dataset["test_r3"]]
            )
        }
    )

    dataset.save_to_disk("anli-phrase-augmented")


if __name__ == "__main__":
    main()
