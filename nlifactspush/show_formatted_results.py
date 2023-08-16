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

def resample_by_column(df, column):
    out = []
    for _, elems in df.groupby(column):
        out.append(elems.sample(frac=1.0, replace=True))

    o = pd.concat(out).reset_index(drop=True)
    return o

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_files", nargs="+", type=Path)
    parser.add_argument("-m", "--mode", choices=["e", "e-c"], default="e-c")
    parser.add_argument("--split-frank-corpus", action="store_true", default=False)
    parser.add_argument("-b", "--bootstrap", type=int, default=0)
    
    args = parser.parse_args()

    all_scores = {}
    for idx, score_file in enumerate(args.score_files):
        score_df = pd.read_csv(score_file)
        all_scores[str(idx) + ": " + fname_to_header(score_file.name)] = score_df
    


    if args.split_frank_corpus:
        FRANK_CNN_DM_START = 296
        first_score = next(iter(all_scores.values()))
        frank_start = (first_score["corpus"] == "frank_dev").idxmax()
        frank_end = (first_score["corpus"].iloc[frank_start:] != "frank_dev").idxmax()
        cnn_dm_start = frank_start + FRANK_CNN_DM_START
        for score in all_scores.values():
            assert score.columns[-1] == "corpus"
            score.iloc[frank_start:cnn_dm_start, -1] = "frank_dev:xsum"
            score.iloc[cnn_dm_start:frank_end, -1] = "frank_dev:cnndm"


    all_results = {}
    all_out_keys = set()
    all_cis = {}
    for key, score_df in all_scores.items():
        score_df = score_df.iloc[:]
        scores = score_df["prediction"].map(lambda x: eval(x))
        if args.mode == "e":
            scores = scores.map(lambda x: x[0])
        elif args.mode == "e-c":
            scores = scores.map(lambda x: x[0] - x[2])
        score_df["scores"] = scores
        
        avg, out = compute_roc_aucs(scores, score_df["label"], score_df["corpus"])
        all_scores = defaultdict(list)
        all_avg = []
        if args.bootstrap != 0:
            for _ in range(args.bootstrap):
                resampled = resample_by_column(score_df, "corpus")
                sample_avg, sample_out = compute_roc_aucs(resampled["scores"], resampled["label"], resampled["corpus"])

                all_avg.append(sample_avg)
                for k, v in sample_out.items():
                    all_scores[k].append(v)

            avg_lower_95_ci = np.percentile(all_avg, 2.5)
            avg_upper_95_ci = np.percentile(all_avg, 97.5)

            out_95_cis = {}

            for corpus_key, vals in all_scores.items():
                out_95_cis[corpus_key] = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
            
            all_cis[key] = ((avg_lower_95_ci, avg_upper_95_ci), out_95_cis)

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

            if args.bootstrap > 0:
                if corpus_key == "__AVG__":
                    lb = all_cis[key][0][0]
                    ub = all_cis[key][0][1]
                else:
                    lb = all_cis[key][1][corpus_key][0]
                    ub = all_cis[key][1][corpus_key][1]

                row[-1] += f" {lb * 100.0:.1f}, {ub * 100.0:.1f}"
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()