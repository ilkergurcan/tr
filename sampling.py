## SUB SAMPLE

import numpy as np
import pandas as pd

def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    n_total: int,
    min_per_class: int = 5,
    seed: int = 42,
):
    # counts per class
    counts = df[label_col].value_counts()
    total = counts.sum()

    # proportional allocation
    take = (counts / total * n_total).round().astype(int)

    # ensure at least min_per_class (but not more than available)
    take = take.clip(lower=min_per_class, upper=counts)

    # adjust to hit n_total exactly (optional but nice)
    diff = int(n_total - take.sum())
    if diff != 0:
        # add/remove from largest classes (that still have room)
        order = counts.index.tolist()
        if diff > 0:
            for c in order:
                if diff == 0: break
                if take[c] < counts[c]:
                    take[c] += 1
                    diff -= 1
        else:
            for c in order:
                if diff == 0: break
                if take[c] > min_per_class:
                    take[c] -= 1
                    diff += 1

    # random rank per row inside each class using a single global shuffle key
    rng = np.random.default_rng(seed)
    u = rng.random(len(df))

    tmp = df[[label_col]].copy()
    tmp["_u"] = u

    # sort by (class, random) then take top N_i per class via cumcount
    sorted_idx = tmp.sort_values([label_col, "_u"]).index
    tmp2 = tmp.loc[sorted_idx]
    tmp2["_rn"] = tmp2.groupby(label_col).cumcount()

    take_df = take.rename("_take").to_frame()
    tmp2 = tmp2.join(take_df, on=label_col)

    sampled_idx = tmp2.index[tmp2["_rn"] < tmp2["_take"]]
    return df.loc[sampled_idx].copy()

# usage
# sample_df = stratified_sample(full_df, label_col="category", n_total=200_000, min_per_class=10)

import pandas as pd

df["text_len"] = df["product_text"].str.len().fillna(0)

# 10 length buckets (quantiles)
df["len_bin"] = pd.qcut(df["text_len"], q=10, duplicates="drop")

# combined strata (category + length bucket)
df["strata"] = df["category"].astype(str) + "||" + df["len_bin"].astype(str)

sample_df = stratified_sample(df, label_col="strata", n_total=20_000, min_per_class=12, seed=42)

# keep original columns, drop helper cols if you want
sample_df = sample_df.drop(columns=["text_len", "len_bin", "strata"])
