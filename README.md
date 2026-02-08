

def eval_leaf_topk(df, mask=None, k_list=(1,3,5), batch=50_000):
    if mask is None:
        mask = np.ones(len(df), dtype=bool)

    sub = df.loc[mask].copy()
    y_str = sub[LEAF_TEST_COL].astype(str).values

    # Map true labels to ids; ignore unknowns (-1)
    y_id = np.array([le_leaf.transform([c])[0] if c in le_leaf.classes_ else -1 for c in y_str], dtype=np.int32)
    valid = y_id >= 0
    sub = sub.iloc[valid]
    y_id = y_id[valid]

    n = len(sub)
    if n == 0:
        return {"n": 0}

    # accumulators
    hits = {k: 0 for k in k_list}
    pred_top1 = np.empty(n, dtype=np.int32)

    # batch loop
    for start in tqdm(range(0, n, batch), desc="Leaf eval (top-k)"):
        end = min(start+batch, n)
        Xb = X_text(sub["hash_text_plus_main2"].iloc[start:end])

        S = clf_leaf.decision_function(Xb)
        if S.ndim == 1:
            # binary edge-case
            S = np.vstack([-S, S]).T

        yb = y_id[start:end]

        # top-1
        p1 = S.argmax(axis=1).astype(np.int32)
        pred_top1[start:end] = p1
        hits[1] += int((p1 == yb).sum())

        # top-k hits
        for k in k_list:
            if k == 1:
                continue
            kk = min(k, S.shape[1])
            idx = np.argpartition(-S, kth=kk-1, axis=1)[:, :kk]
            hits[k] += int(np.any(idx == yb[:, None], axis=1).sum())

    # metrics on top-1 predictions
    out = {
        "n": int(n),
        "top1_acc": hits[1] / n,
        "macro_f1": f1_score(y_id, pred_top1, average="macro"),
        "micro_f1": f1_score(y_id, pred_top1, average="micro"),
        "weighted_f1": f1_score(y_id, pred_top1, average="weighted"),
    }
    for k in k_list:
        out[f"top{k}_acc"] = hits[k] / n
    return out

# ALL / OVERLAP / NON-OVERLAP
mask_all = np.ones(len(test_df), dtype=bool)
mask_ov  = test_df["is_exact_leak"].values
mask_no  = ~mask_ov

res_all = eval_leaf_topk(test_df, mask_all)
res_ov  = eval_leaf_topk(test_df, mask_ov)
res_no  = eval_leaf_topk(test_df, mask_no)

metrics_topk = pd.DataFrame([
    {"split":"ALL", **res_all},
    {"split":"OVERLAP", **res_ov},
    {"split":"NON-OVERLAP", **res_no},
])

display(metrics_topk)
