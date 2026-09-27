"""
Stage 4 - score the test candidates and write the two submission files.

Produces, in ``student_resource/output/``:

    matching_results.tsv   final matches (the leaderboard file)
    candidate_pairs.tsv    the candidate set stage 1 handed to the model

Both must contain **exactly one row per Source-1 test entity, in the test file's own
order**, including entities for which blocking found nothing and entities in France -
a country absent from training. Rather than trusting an in-memory mapping of 1.7M ids,
the writer streams ``test_source1.tsv`` itself and emits one row per line read. That
makes "every entity appears exactly once" a property of the loop instead of something
to verify afterwards.

The country shards written by stage 0 preserve, within each country, the order of the
original file. Rows are therefore consumed with a per-country cursor, and each one is
asserted against the streamed ``entity_id`` - if stage 0 ever stops preserving order
this fails loudly instead of silently mislabelling every match in the submission.

Scoring is chunked over the memmapped feature matrix so that peak memory stays at one
chunk regardless of how large the test candidate set is.
"""

import argparse
import csv
import json
import os

import numpy as np

from common import CONFIG, SOURCE_FILES, data_path, log, rss_gb, work_path
from s1_block import countries, load_shard
from s2_features import feature_path

SCORE_CHUNK = 2_000_000


def predict_country(booster, country, thr, margin):
    """Return (matched_lists, candidate_lists), one entry per Source-1 row of the shard."""
    s1 = load_shard("test", country, "s1", columns=["entity_id"])
    n1 = len(s1["entity_id"])
    cand_path = work_path("s1", f"test_{country}_cand.npz")
    matches = [""] * n1
    cands = [""] * n1
    if not os.path.exists(cand_path):
        log(f"  {country}: no candidate file - {n1:,} entities emitted empty")
        return s1["entity_id"], matches, cands

    z = np.load(cand_path, allow_pickle=False)
    s1_idx = z["s1_idx"].astype(np.int64)
    ref_idx = z["ref_idx"].astype(np.int64)
    del z
    ref = load_shard("test", country, "ref", columns=["entity_id"])
    ref_ids = ref["entity_id"]
    del ref

    X = np.load(feature_path("test", country), mmap_mode="r")
    n_pairs = len(s1_idx)
    assert X.shape[0] == n_pairs, f"{country}: features/candidates length mismatch"
    prob = np.empty(n_pairs, dtype=np.float32)
    for s in range(0, n_pairs, SCORE_CHUNK):
        e = min(s + SCORE_CHUNK, n_pairs)
        prob[s:e] = booster.predict(np.asarray(X[s:e]))
        log(f"    {country}: scored {e:,}/{n_pairs:,} [rss {rss_gb():.2f}GB]")
    del X

    # s1_idx is sorted ascending by construction in stage 1.
    bounds = np.searchsorted(s1_idx, np.arange(n1 + 1))
    n_matched = 0
    for i in range(n1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        rr = ref_idx[lo:hi]
        cands[i] = ",".join(ref_ids[rr].tolist())
        p = prob[lo:hi]
        ok = p >= thr
        if ok.any():
            ok &= p >= (p.max() - margin)
            sel = rr[ok]
            if len(sel):
                matches[i] = ",".join(ref_ids[sel].tolist())
                n_matched += 1
    log(f"  {country}: {n_matched:,}/{n1:,} entities got >=1 match "
        f"[rss {rss_gb():.2f}GB]")
    return s1["entity_id"], matches, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=CONFIG["out_dir"])
    ap.add_argument("--countries", default="")
    args = ap.parse_args()

    import lightgbm as lgb

    booster = lgb.Booster(model_file=work_path("s3", "model.txt"))
    with open(work_path("s3", "decision.json")) as f:
        dec = json.load(f)
    thr, margin = dec["thr"], dec["margin"]
    log(f"=== stage 4: thr={thr:.2f} margin={margin:.2f} "
        f"(holdout F0.5 {dec.get('holdout_f05', float('nan')):.4f}) ===")

    only = {c for c in args.countries.split(",") if c}
    cs = [c for c in countries("test") if not only or c in only]

    shard_ids, shard_match, shard_cand, cursor = {}, {}, {}, {}
    for c in cs:
        ids, m, k = predict_country(booster, c, thr, margin)
        shard_ids[c], shard_match[c], shard_cand[c], cursor[c] = ids, m, k, 0

    os.makedirs(args.out_dir, exist_ok=True)
    mpath = os.path.join(args.out_dir, "matching_results.tsv")
    cpath = os.path.join(args.out_dir, "candidate_pairs.tsv")
    src1 = data_path(SOURCE_FILES[("test", 1)])

    n = 0
    with open(src1, encoding="utf-8", newline="") as f, \
         open(mpath, "w", encoding="utf-8", newline="") as fm, \
         open(cpath, "w", encoding="utf-8", newline="") as fc:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for row in reader:
            if not row:
                continue
            eid = row[0]
            country = row[3] if len(row) > 3 else ""
            if country in cursor:
                i = cursor[country]
                assert shard_ids[country][i] == eid, (
                    f"shard order drift for {country} at row {i}: "
                    f"{shard_ids[country][i]!r} != {eid!r}")
                cursor[country] = i + 1
                fm.write(f"{eid}\t{shard_match[country][i]}\n")
                fc.write(f"{eid}\t{shard_cand[country][i]}\n")
            else:
                # A country with no prepared shard still owes exactly one row.
                fm.write(f"{eid}\t\n")
                fc.write(f"{eid}\t\n")
            n += 1
    for c in cs:
        assert cursor[c] == len(shard_ids[c]), f"{c}: {cursor[c]} of {len(shard_ids[c])} rows consumed"
    log(f"wrote {n:,} rows -> {mpath}")
    log(f"wrote {n:,} rows -> {cpath}")
    log("stage 4 complete")


if __name__ == "__main__":
    main()
