"""
Stage 2 - pairwise feature construction.

Turns each candidate pair into a fixed-width float32 feature vector. The scale here is
the design constraint: the test candidate set is ~100M pairs, so nothing may be
computed with a Python loop over pairs. Every string similarity goes through
``rapidfuzz.process.cpdist``, which evaluates two equal-length sequences element-wise in
multithreaded C++.

MEMORY: the feature matrix is written straight into a ``np.memmap`` and filled a chunk
at a time. Held in RAM, test/India alone would be ~3GB and the whole test split ~11GB,
which does not fit beside the shards on a 16GB machine. Writing through a memmap keeps
the resident cost at one chunk (~0.3GB) and leaves the OS to flush pages.

Feature groups
--------------
* Retrieval scores from all three stage-1 indexes, kept separate. Which index found a
  pair is itself informative: a pair retrieved by the name index but not the address
  index is a different kind of candidate from one retrieved by both.
* Name similarity, four different scorers. They disagree in useful ways: ``ratio`` is
  sensitive to typos, ``token_sort_ratio`` absorbs word-order transposition (a
  documented noise pattern here), ``token_set_ratio`` absorbs extra/missing words such
  as dropped legal suffixes, and ``partial_ratio`` catches a name contained inside a
  longer one ("Upper Cascade Hall" inside "Upper Cascade Hall Group").
* Skeleton similarity on vowel-stripped names, which is what lines up romanized Indic
  names with their Latin counterparts.
* Address similarity plus digit agreement. House numbers and PIN codes survive almost
  every kind of textual noise in this data, so they are compared separately from the
  address text.
* Blocking context: the cosine, its rank within the entity, and how far it sits below
  the best candidate for that entity.
* Competition features. Ground truth is a strict many-to-one mapping - every link
  points to a distinct Source-2/3 record, so no reference record is ever shared between
  two Source-1 entities. A candidate that is also some other entity's best match is
  therefore much less likely to be a true match here, and these features expose that.

Output per (split, country):
    <work>/s2/<split>_<country>_X.npy   float32 (n_pairs, n_features)  memmapped
    <work>/s2/<split>_<country>_y.npy   int8    (n_pairs,)   train only
"""

import os

import numpy as np
from rapidfuzz import fuzz, process

from common import log, read_ground_truth, rss_gb, work_path
from s1_block import countries, load_shard

FEATURES = [
    "cos_blob", "cos_addr", "cos_name", "cos_max", "n_idx_hit",
    "cos_rank", "cos_best", "cos_margin", "n_cands",
    "name_ratio", "name_tsort", "name_tset", "name_partial",
    "skel_ratio", "skel_partial",
    "addr_ratio", "addr_tsort", "addr_tset",
    "blob_tset", "digits_tset",
    "name_len_ratio", "addr_len_ratio",
    "ref_addr_empty", "s1_addr_empty", "ref_name_len", "s1_name_len",
    "src_is_s3",
    "ref_best_cos", "ref_cos_margin", "ref_n_s1",
]
PAIR_CHUNK = 1_000_000  # pairs scored per rapidfuzz call; bounds transient memory


def _cpdist(a, b, scorer):
    """Element-wise similarity for two equal-length string sequences, 0-100 scale."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _safe_ratio(a, b):
    """min/max of two length arrays, defined as 0 when both are zero.

    Two records whose names both normalize to the empty string would otherwise produce
    0/0 = NaN, and a single NaN feature silently poisons a LightGBM split.
    """
    hi = np.maximum(a, b)
    return np.where(hi > 0, np.minimum(a, b) / np.maximum(hi, 1.0), 0.0)


def _group_stats(key, val, n_groups):
    """Per-group max and count for a key array that is *not* assumed sorted."""
    best = np.zeros(n_groups, dtype=np.float32)
    np.maximum.at(best, key, val)
    counts = np.bincount(key, minlength=n_groups).astype(np.float32)
    return best, counts


def feature_path(split, country):
    return work_path("s2", f"{split}_{country}_X.npy")


def build_features(split, country):
    cand_path = work_path("s1", f"{split}_{country}_cand.npz")
    if not os.path.exists(cand_path):
        log(f"  {split}/{country}: no candidates, skipped")
        return
    z = np.load(cand_path, allow_pickle=False)
    s1_idx, ref_idx = z["s1_idx"], z["ref_idx"]
    cos_blob, cos_addr, cos_name = z["cos_blob"], z["cos_addr"], z["cos_name"]
    sample_idx = z["sample_idx"]
    del z

    s1 = load_shard(split, country, "s1")
    ref = load_shard(split, country, "ref")
    if len(sample_idx):
        s1 = {k: v[sample_idx] for k, v in s1.items()}
    n1 = len(s1["entity_id"])
    n_ref = len(ref["entity_id"])
    n_pairs = len(s1_idx)
    log(f"  {split}/{country}: {n_pairs:,} pairs over {n1:,} S1 entities "
        f"[rss {rss_gb():.2f}GB]")
    if n_pairs == 0:
        return

    s1_idx = s1_idx.astype(np.int64, copy=False)
    ref_idx = ref_idx.astype(np.int64, copy=False)

    # A single scalar retrieval score for the ranking/context features. Max rather than
    # sum: the indexes have different score scales and a pair found by one strong index
    # should not be penalised for being missed by the others.
    cos = np.maximum.reduce([cos_blob, cos_addr, cos_name])
    n_idx_hit = ((cos_blob > 0).astype(np.float32) + (cos_addr > 0)
                 + (cos_name > 0)).astype(np.float32)

    # --- per-entity and per-reference blocking context -------------------------------
    # cos_rank: position of this candidate among the entity's candidates by cosine.
    order = np.lexsort((-cos, s1_idx))
    rank = np.empty(n_pairs, dtype=np.float32)
    starts = np.searchsorted(s1_idx[order], np.arange(n1), side="left")
    rank[order] = (np.arange(n_pairs) - starts[s1_idx[order]]).astype(np.float32)
    del order, starts

    cos_best, n_cands = _group_stats(s1_idx, cos, n1)
    # Competition on the reference side: the many-to-one structure means a reference
    # record claimed strongly by some other entity is unlikely to belong to this one.
    ref_best, ref_n = _group_stats(ref_idx, cos, n_ref)

    X = np.lib.format.open_memmap(
        feature_path(split, country), mode="w+", dtype=np.float32,
        shape=(n_pairs, len(FEATURES)))
    col = {name: i for i, name in enumerate(FEATURES)}

    X[:, col["cos_blob"]] = cos_blob
    X[:, col["cos_addr"]] = cos_addr
    X[:, col["cos_name"]] = cos_name
    X[:, col["cos_max"]] = cos
    X[:, col["n_idx_hit"]] = n_idx_hit
    X[:, col["cos_rank"]] = rank
    X[:, col["cos_best"]] = cos_best[s1_idx]
    X[:, col["cos_margin"]] = cos - cos_best[s1_idx]
    X[:, col["n_cands"]] = n_cands[s1_idx]
    X[:, col["ref_best_cos"]] = ref_best[ref_idx]
    X[:, col["ref_cos_margin"]] = cos - ref_best[ref_idx]
    X[:, col["ref_n_s1"]] = ref_n[ref_idx]
    X[:, col["src_is_s3"]] = (ref["src"][ref_idx] == 3).astype(np.float32)
    del cos_best, n_cands, ref_best, ref_n, rank, n_idx_hit
    del cos_blob, cos_addr, cos_name, cos

    s1_name, s1_addr = s1["name_norm"], s1["addr_norm"]
    s1_skel, s1_dig = s1["skel"], s1["digits"]
    r_name, r_addr = ref["name_norm"], ref["addr_norm"]
    r_skel, r_dig = ref["skel"], ref["digits"]

    for start in range(0, n_pairs, PAIR_CHUNK):
        stop = min(start + PAIR_CHUNK, n_pairs)
        qi, ri = s1_idx[start:stop], ref_idx[start:stop]
        an = s1_name[qi].tolist(); bn = r_name[ri].tolist()
        aa = s1_addr[qi].tolist(); ba = r_addr[ri].tolist()
        ak = s1_skel[qi].tolist(); bk = r_skel[ri].tolist()
        ad = s1_dig[qi].tolist(); bd = r_dig[ri].tolist()

        X[start:stop, col["name_ratio"]] = _cpdist(an, bn, fuzz.ratio)
        X[start:stop, col["name_tsort"]] = _cpdist(an, bn, fuzz.token_sort_ratio)
        X[start:stop, col["name_tset"]] = _cpdist(an, bn, fuzz.token_set_ratio)
        X[start:stop, col["name_partial"]] = _cpdist(an, bn, fuzz.partial_ratio)
        X[start:stop, col["skel_ratio"]] = _cpdist(ak, bk, fuzz.ratio)
        X[start:stop, col["skel_partial"]] = _cpdist(ak, bk, fuzz.partial_ratio)
        X[start:stop, col["addr_ratio"]] = _cpdist(aa, ba, fuzz.ratio)
        X[start:stop, col["addr_tsort"]] = _cpdist(aa, ba, fuzz.token_sort_ratio)
        X[start:stop, col["addr_tset"]] = _cpdist(aa, ba, fuzz.token_set_ratio)
        X[start:stop, col["digits_tset"]] = _cpdist(ad, bd, fuzz.token_set_ratio)
        del ak, bk, ad, bd

        ab = [f"{n} {a}" for n, a in zip(an, aa)]
        bb = [f"{n} {a}" for n, a in zip(bn, ba)]
        X[start:stop, col["blob_tset"]] = _cpdist(ab, bb, fuzz.token_set_ratio)
        del ab, bb

        m = stop - start
        la = np.fromiter((len(x) for x in an), np.float32, m)
        lb = np.fromiter((len(x) for x in bn), np.float32, m)
        ra = np.fromiter((len(x) for x in aa), np.float32, m)
        rb = np.fromiter((len(x) for x in ba), np.float32, m)
        del an, bn, aa, ba
        X[start:stop, col["name_len_ratio"]] = _safe_ratio(la, lb)
        X[start:stop, col["addr_len_ratio"]] = _safe_ratio(ra, rb)
        X[start:stop, col["ref_addr_empty"]] = (rb == 0).astype(np.float32)
        X[start:stop, col["s1_addr_empty"]] = (ra == 0).astype(np.float32)
        X[start:stop, col["ref_name_len"]] = lb
        X[start:stop, col["s1_name_len"]] = la
        del la, lb, ra, rb
        log(f"    {split}/{country}: features {stop:,}/{n_pairs:,} "
            f"[rss {rss_gb():.2f}GB]")

    X.flush()
    del X

    if split == "train":
        gt = read_ground_truth()
        s1_ids = s1["entity_id"]
        ref_ids = ref["entity_id"]
        truth_sets = [set(gt.get(str(i), ())) for i in s1_ids]
        y = np.fromiter(
            (1 if ref_ids[r] in truth_sets[q] else 0 for q, r in zip(s1_idx, ref_idx)),
            dtype=np.int8, count=n_pairs)
        np.save(work_path("s2", f"{split}_{country}_y.npy"), y)
        log(f"  {split}/{country}: positives={int(y.sum()):,} / {n_pairs:,} "
            f"({100*y.mean():.2f}%)")
    log(f"  {split}/{country}: features written [rss {rss_gb():.2f}GB]")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=["train", "test"])
    ap.add_argument("--countries", default="", help="comma-separated subset")
    args = ap.parse_args()
    only = {c for c in args.countries.split(",") if c}
    for split in args.splits:
        log(f"=== features {split} ===")
        for c in countries(split):
            if only and c not in only:
                continue
            build_features(split, c)
    log("stage 2 complete")


if __name__ == "__main__":
    main()
