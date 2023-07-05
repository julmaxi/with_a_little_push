# Code for the 2023 ACL Short Paper "With a Little Push, NLI Models _can_ Robustly and Efficiently Predict Faithfulness"

## Paper

For more details read the [Paper](https://arxiv.org/pdf/2305.16819)

## Usage

All relevant code can be found in the ``nlifactspush`` module.
To train a new augmented model, first run ``augment_dataset.py``, followed by ``train.py``.
To derive scores, use ``score.py``. The latter expects to receive the dataset (e.g. [TRUE](https://github.com/google-research/true)) in a jsonl format. Each instance should have the following fields:

- ``label``: A binary faithfulness label
- ``corpus``: Used for grouping results
- ``grounding``: The grounding you want to check faithfulness on
- ``generation``: The generation to score
