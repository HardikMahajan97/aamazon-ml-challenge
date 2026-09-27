"""
Stage 3 - train the pairwise matcher and tune the decision rule for F_0.5.

A LightGBM binary classifier scores each candidate pair. The model itself is the easy
part; the decision rule on top of it is where the metric is won or lost, so this stage
spends most of its effort there.

Why a GBDT and not a cross-encoder: the features are ~30 dense numeric similarities
with strong monotone structure, the candidate set is ~100M pairs at inference, and the
challenge caps models at 8B parameters under MIT/Apache. A gradient-boosted tree
ensemble scores 100M pairs in minutes on CPU and is trivially licensable; a transformer
re-ranker would cost orders of magnitude more for a metric that is dominated by
threshold placement rather than by subtle semantics.

Sampling
--------
Negatives outnumber positives ~20:1 in the candidate set. The *training* split
downsamples negatives (all positives kept, a fixed fraction of negatives) purely to fit
the fit in memory. The *holdout* split is never downsampled: precision must be measured
against the true negative density, or the tuned threshold will be far too permissive.
``scale_pos_weight`` is left at 1 and the raw probability is calibrated by the
threshold sweep instead.

Decision rule
-------------
F_0.5 is macro-averaged per Source-1 entity and a singleton scores a full 1.0 for an
empty prediction, so the rule is tuned on entity-level score, not pair-level F1:

* ``thr``      - minimum probability for a candidate to be emitted at all.
* ``margin``   - a candidate must also be within this much of the entity's best
                 probability. This is what suppresses the long tail of
                 near-duplicate business names that share a street, which is the
                 dominant false-merge mode in this data.

Both are swept jointly on the holdout and the best pair is written to the model
bundle for stage 4 to apply unchanged.

Output:
    <work>/s3/model.txt        LightGBM booster
    <work>/s3/decision.json    {"thr": float, "margin": float, "features": [...]}
"""

import argparse
import json
import os

import numpy as np

from common import CONFIG, f_beta, log, read_ground_truth, rss_gb, work_path
from s1_block import countries, load_shard
from s2_features import FEATURES, feature_path


def _load_country(split, country):
    """Return (X memmap, y, s1_row, ref_entity_ids, s1_entity_ids) for one country."""
    cand = work_path("s1", f"{split}_{country}_cand.npz")
    xpath = feature_path(split, country)
    ypath = work_path("s2", f"{split}_{country}_y.npy")
    if not (os.path.exists(cand) and os.path.exists(xpath) and os.path.exists(ypath)):
        return None
    z = np.load(cand, allow_pickle=False)
    s1_idx, ref_idx, sample_idx = z["s1_idx"], z["ref_idx"], z["sample_idx"]
    del z
    X = np.load(xpath, mmap_mode="r")
    y = np.load(ypath)
    s1 = load_shard(split, country, "s1", columns=["entity_id"])
    ref = load_shard(split, country, "ref", columns=["entity_id"])
    s1_ids = s1["entity_id"]
    if len(sample_idx):
        s1_ids = s1_ids[sample_idx]
    return X, y, s1_idx.astype(np.int64), ref["entity_id"][ref_idx.astype(np.int64)], s1_ids


def _split_entities(n1, holdout_frac, seed):
    """Deterministic entity-level train/holdout mask. Splitting by *pair* would leak a
    business across both sides and make the tuned threshold optimistic."""
    rng = np.random.default_rng(seed)
    return rng.random(n1) < holdout_frac


def assemble(split, cs, neg_frac, holdout_frac, seed):
    tr_X, tr_y = [], []
    ho = []   # per country: (X_rows, y, entity_index, ref_ids, s1_ids)
    for c in cs:
        got = _load_country(split, c)
        if got is None:
            log(f"  {c}: missing stage-1/2 artefacts, skipped")
            continue
        X, y, s1_idx, ref_ids, s1_ids = got
        n1 = len(s1_ids)
        is_ho = _split_entities(n1, holdout_frac, seed)
        pair_ho = is_ho[s1_idx]

        # --- training side: all positives, a sample of negatives ---------------------
        rng = np.random.default_rng(seed + 1)
        keep = (~pair_ho) & ((y == 1) | (rng.random(len(y)) < neg_frac))
        idx = np.flatnonzero(keep)
        tr_X.append(np.asarray(X[idx]))
        tr_y.append(y[idx])
        log(f"  {c}: train pairs={len(idx):,} (pos={int(y[idx].sum()):,})  "
            f"[rss {rss_gb():.2f}GB]")

        # --- holdout side: every pair, no downsampling --------------------------------
        hidx = np.flatnonzero(pair_ho)
        ho.append((np.asarray(X[hidx]), y[hidx], s1_idx[hidx], ref_ids[hidx],
                   s1_ids, is_ho))
        log(f"  {c}: holdout pairs={len(hidx):,} over {int(is_ho.sum()):,} entities")
        del X, y, ref_ids
    if not tr_X:
        raise SystemExit("no training data found - run stages 1 and 2 first")
    return np.concatenate(tr_X), np.concatenate(tr_y), ho


def tune_decision(prob, ho, gt):
    """Joint sweep of (threshold, margin), scored with the real macro F_0.5.

    Every holdout entity contributes a term, including those with no candidates at all
    and those whose ground truth is empty - exactly as the leaderboard scores it.
    """
    # Flatten all holdout countries into one pair list plus a global entity index.
    s1g, refs = [], []
    truths = []       # per global entity: set of true ids
    base = 0
    for (_X, _y, s1_idx, ref_ids, s1_ids, is_ho) in ho:
        ho_rows = np.flatnonzero(is_ho)
        remap = np.full(len(s1_ids), -1, dtype=np.int64)
        remap[ho_rows] = np.arange(len(ho_rows)) + base
        s1g.append(remap[s1_idx])
        refs.append(ref_ids)
        truths.extend(set(gt.get(str(i), ())) for i in s1_ids[ho_rows])
        base += len(ho_rows)
    n_ent = base
    s1g = np.concatenate(s1g)
    refs = np.concatenate(refs)
    assert len(prob) == len(s1g)

    order = np.argsort(s1g, kind="stable")
    s1g, refs, prob = s1g[order], refs[order], prob[order]
    bounds = np.searchsorted(s1g, np.arange(n_ent + 1))

    best = (-1.0, 0.5, 1.0)
    for thr in np.arange(0.30, 0.96, 0.05):
        for margin in (1.01, 0.5, 0.35, 0.25, 0.15):
            total = 0.0
            for e in range(n_ent):
                lo, hi = bounds[e], bounds[e + 1]
                truth = truths[e]
                if hi > lo:
                    p = prob[lo:hi]
                    ok = p >= thr
                    if ok.any():
                        ok &= p >= (p.max() - margin)
                    pred = set(refs[lo:hi][ok].tolist())
                else:
                    pred = set()
                if not truth and not pred:
                    total += 1.0
                    continue
                if not truth or not pred:
                    continue
                tp = len(pred & truth)
                if tp:
                    total += f_beta(tp / len(pred), tp / len(truth))
            score = total / max(n_ent, 1)
            if score > best[0]:
                best = (score, float(thr), float(margin))
            log(f"    thr={thr:.2f} margin={margin:.2f} -> macro F0.5 = {score:.4f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-frac", type=float, default=0.25)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--countries", default="")
    args = ap.parse_args()

    import lightgbm as lgb

    only = {c for c in args.countries.split(",") if c}
    cs = [c for c in countries("train") if not only or c in only]
    log(f"=== stage 3: training on countries={cs} ===")
    Xtr, ytr, ho = assemble("train", cs, args.neg_frac, args.holdout_frac,
                            CONFIG["seed"])
    log(f"  train matrix: {Xtr.shape}  positives={int(ytr.sum()):,} "
        f"[rss {rss_gb():.2f}GB]")

    Xho = np.concatenate([h[0] for h in ho])
    yho = np.concatenate([h[1] for h in ho])

    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, num_threads=os.cpu_count(),
                  seed=CONFIG["seed"])
    ds_tr = lgb.Dataset(Xtr, label=ytr, feature_name=list(FEATURES))
    ds_ho = lgb.Dataset(Xho, label=yho, feature_name=list(FEATURES), reference=ds_tr)
    booster = lgb.train(params, ds_tr, num_boost_round=args.rounds,
                        valid_sets=[ds_ho], valid_names=["holdout"],
                        callbacks=[lgb.early_stopping(40, verbose=False),
                                   lgb.log_evaluation(50)])
    log(f"  best iteration: {booster.best_iteration}")
    del Xtr, ytr, ds_tr, ds_ho

    imp = sorted(zip(FEATURES, booster.feature_importance("gain")),
                 key=lambda t: -t[1])
    log("  top features by gain: " + ", ".join(f"{n}={g:.0f}" for n, g in imp[:10]))

    prob = booster.predict(Xho, num_iteration=booster.best_iteration)
    del Xho, yho
    gt = read_ground_truth()
    log("  tuning decision rule on holdout ...")
    score, thr, margin = tune_decision(prob, ho, gt)
    log(f"  BEST: macro F0.5 = {score:.4f}  (thr={thr:.2f}, margin={margin:.2f})")

    booster.save_model(work_path("s3", "model.txt"),
                       num_iteration=booster.best_iteration)
    with open(work_path("s3", "decision.json"), "w") as f:
        json.dump({"thr": thr, "margin": margin, "holdout_f05": score,
                   "features": list(FEATURES)}, f, indent=2)
    log("stage 3 complete")


if __name__ == "__main__":
    main()
