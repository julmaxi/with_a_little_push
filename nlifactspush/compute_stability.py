import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
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
    parser.add_argument("-m", "--mode", choices=["e", "e-c"], default="e-c")

    args = parser.parse_args()

    all_scores = []
    for score_file in args.score_files:
        score_df = pd.read_csv(score_file)
        all_scores.append(score_df)
    
    all_results = defaultdict(list)

    for score in all_scores:
        scores = score["prediction"].map(lambda x: eval(x))

        if args.mode == "e":
            scores = scores.map(lambda x: x[0])
        elif args.mode == "e-c":
            scores = scores.map(lambda x: x[0] - x[2])
        avg, out = compute_roc_aucs(scores, score["label"], score["corpus"])
        for k, v in out.items():
            all_results[k].append(v)
        all_results["__AVG__"].append(avg)
    
    for k, v in sorted(all_results.items()):
        print(f"{k}:\t{np.std(v) * 100.0:.1f}\t{sum(v) / len(v) * 100.0:.1f}\t{min(v) * 100.0:.1f}\t{max(v) * 100.0:.1f}")


if __name__ == "__main__":
    main()