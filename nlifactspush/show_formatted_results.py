import argparse
import pandas as pd
from pathlib import Path
from .score import compute_roc_aucs


def fname_to_header(name):
    config, train_info = name.split("-", 1)
    return config.rsplit("_", 1)[0]


def print_table(table):
    col_max_width = [0 for _ in table[0]]

    for row in table:
        for i, col in enumerate(row):
            col_max_width[i] = max(col_max_width[i], len(col))
    
    padded_table = []
    for row in table:
        padded_row = []
        for i, col in enumerate(row):
            padded_row.append(col.ljust(col_max_width[i] + 1))
        padded_table.append(padded_row)

    print("\n".join(["\t".join(row) for row in padded_table]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_files", nargs="+", type=Path)

    args = parser.parse_args()

    all_scores = {}
    for score_file in args.score_files:
        score_df = pd.read_csv(score_file)
        all_scores[fname_to_header(score_file.name)] = score_df
    
    all_results = {}

    all_out_keys = set()

    for key, score in all_scores.items():
        #scores = score["prediction"].map(lambda x: eval(x)[0])
        scores = score["scores"]
        avg, out = compute_roc_aucs(scores, score["label"], score["corpus"])
        all_out_keys.update(out)
        all_results[key] = (avg, out)


    all_result_items = sorted(all_results.items())
    table = [["Corpus"] + [k for k, _ in all_result_items]]

    for corpus_key in sorted(all_out_keys) + ["__AVG__"]:
        row = [corpus_key]
        for key, (avg, out) in all_result_items:
            if corpus_key == "__AVG__":
                row.append(f"{avg * 100.0:.1f}")
            else:
                row.append(f"{out[corpus_key] * 100.0:.1f}")
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()