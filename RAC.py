import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# -------------------
# Speed knobs
# -------------------
QUERY_BATCH   = 2048      # query embedding batch
EMB_BATCH     = 1024      # embedder internal batch
K_RETRIEVE    = 200
K_RERANK      = 15        # BIG win: 10-20 recommended
RERANK_BATCH  = 256       # tune to GPU, 128/256/512

# early-exit thresholds (tune)
EARLY_SIM_TH  = 0.85
EARLY_GAP_TH  = 0.05

DECIDE_MODE   = "agg_max"   # or "agg_sumexp" / "top1_neighbor"
TOPK_LIST     = [3, 5]      # keep 1 out to avoid overwriting top1 string


def aggregate_label_scores_sorted(cand_labels_sorted, cand_scores_sorted, mode="agg_max"):
    # cand_labels_sorted: (K,) labels already sorted by score desc
    # cand_scores_sorted: (K,) scores sorted desc
    if mode == "top1_neighbor":
        return [cand_labels_sorted[0]]

    # aggregate per label
    out = {}
    if mode == "agg_max":
        for lab, sc in zip(cand_labels_sorted, cand_scores_sorted):
            sc = float(sc)
            prev = out.get(lab)
            if prev is None or sc > prev:
                out[lab] = sc
    elif mode == "agg_sumexp":
        for lab, sc in zip(cand_labels_sorted, cand_scores_sorted):
            out[lab] = out.get(lab, 0.0) + float(np.exp(sc))
    else:
        raise ValueError("Unknown mode")

    ranked = sorted(out.items(), key=lambda x: -x[1])
    return [x[0] for x in ranked]


def predict_retrieve_rerank_unique(df, text_col="text_for_model", key_col="canon_text"):
    # 1) unique texts
    uniq = df.drop_duplicates(key_col).copy()
    texts = uniq[text_col].fillna("").astype(str).to_numpy()
    N = len(texts)

    pred_top1 = np.empty(N, dtype=object)
    pred_topk = {k: [None]*N for k in TOPK_LIST}
    conf_top1 = np.empty(N, dtype=np.float32)  # optional: store p-like score proxy

    for q0 in tqdm(range(0, N, QUERY_BATCH), desc="Retrieve→Rerank (unique)"):
        q1 = min(q0 + QUERY_BATCH, N)
        q_texts = texts[q0:q1]

        # embed queries
        q_emb = embedder.encode(
            q_texts,
            batch_size=EMB_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        # retrieve
        D, I = index.search(q_emb, K_RETRIEVE)   # D: (b,K)
        b = I.shape[0]

        # early-exit mask (skip rerank)
        top_sim = D[:, 0]
        gap_sim = D[:, 0] - D[:, 1]
        early = (top_sim >= EARLY_SIM_TH) & (gap_sim >= EARLY_GAP_TH)

        # fill early predictions
        if early.any():
            idxs = np.where(early)[0]
            top_idx = I[idxs, 0]
            pred_top1[q0+idxs] = train_labels[top_idx]
            conf_top1[q0+idxs] = top_sim[idxs].astype(np.float32)

            for k in TOPK_LIST:
                # just use retrieved labels for top-k as a fallback
                pred_topk[k] = pred_topk[k]  # keep dict
                labs = train_labels[I[idxs, :k]]
                for j, rr in enumerate(idxs):
                    pred_topk[k][q0+rr] = list(labs[j])

        # rerank only hard cases
        hard_rows = np.where(~early)[0]
        if len(hard_rows) == 0:
            continue

        I_top = I[hard_rows, :K_RERANK]
        hard_texts = q_texts[hard_rows]
        bh = I_top.shape[0]

        # build flat query/cand arrays (no big zip list)
        pair_q = np.repeat(hard_texts, K_RERANK)
        pair_c = train_texts[I_top.reshape(-1)]

        # reranker scoring in batches
        scores = np.empty(len(pair_q), dtype=np.float32)
        for p0 in range(0, len(pair_q), RERANK_BATCH):
            p1 = min(p0 + RERANK_BATCH, len(pair_q))
            # sentence-transformers CrossEncoder supports list of (q,c).
            # Building per-batch list is OK and avoids a 50M list upfront.
            batch_pairs = list(zip(pair_q[p0:p1].tolist(), pair_c[p0:p1].tolist()))
            s = reranker.predict(batch_pairs)  # should run on GPU if configured
            scores[p0:p1] = np.asarray(s, dtype=np.float32)

        scores = scores.reshape(bh, K_RERANK)

        # decide per query
        for local_i, row_i in enumerate(hard_rows):
            cand_idx = I_top[local_i]
            cand_labs = train_labels[cand_idx]
            cand_sco  = scores[local_i]

            order = np.argsort(-cand_sco)
            cand_labs = cand_labs[order]
            cand_sco  = cand_sco[order]

            ranked_labels = aggregate_label_scores_sorted(cand_labs, cand_sco, mode=DECIDE_MODE)
            if len(ranked_labels) == 0:
                ranked_labels = [cand_labs[0]]

            pred_top1[q0 + row_i] = ranked_labels[0]
            conf_top1[q0 + row_i] = float(cand_sco[0])

            for k in TOPK_LIST:
                pred_topk[k][q0 + row_i] = ranked_labels[:k] if len(ranked_labels) >= k else ranked_labels

    # merge back to original df
    uniq["pred_rac_top1"] = pred_top1
    uniq["conf_rac_top1"] = conf_top1
    for k in TOPK_LIST:
        uniq[f"pred_rac_top{k}"] = pred_topk[k]

    out = df[[key_col]].merge(
        uniq[[key_col, "pred_rac_top1", "conf_rac_top1"] + [f"pred_rac_top{k}" for k in TOPK_LIST]],
        on=key_col,
        how="left"
    )
    return out
