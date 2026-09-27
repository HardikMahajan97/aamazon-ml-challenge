"""
Stage 1 - candidate generation: retrieve a generous pool, then rerank it down.

Reduces ~1.7M x 10M potential comparisons to a few dozen candidates per Source-1
entity. The structure is retrieve-then-rerank, which measurement on the real ground
truth showed is necessary - a single index cannot do this job:

Retrieval runs three independent character n-gram TF-IDF indexes over the reference
records of one country and unions their top-k lists. Measured pair recall on 5,000
Indian training entities:

    address index alone   83.96%
    name+address blob     90.12%
    name index alone      57.87%
    union of all three    94.33%

The three are kept separate rather than concatenated because either field alone can be
the only usable signal: ~3.3% of reference records have an empty address, and ~7% carry
an Indic-script name whose romanization is imperfect. Concatenating them into one
vector also lets a long, repetitive address (Indian addresses repeat city and state
tokens heavily) swamp a short discriminative name - tested directly, and a
length-balanced name/address vector scored *worse* (85.99%) than the plain blob, so the
union of separate indexes is what actually works.

Reranking then cuts the ~460-candidate pool to the ~60 the model will score. Ranking by
``name_sim + addr_sim + cos_blob + cos_addr + cos_name`` retains 91.5% of true links at
60 candidates, against a 94.33% pool ceiling; simpler rules are markedly worse (a plain
0.5*name + 0.5*address average keeps only 89.7%).

Two implementation details carry most of the performance:

* High-document-frequency n-grams are pruned outright. Without it every query touches
  almost the whole reference set through n-grams like " rue" or "ville": measured on
  400k French references, an unpruned query produced 323,109 non-zero similarities
  versus 1,416 once common n-grams were dropped.
* Query chunks are processed by a forked worker pool. scipy's sparse product is
  single-threaded, and the reference matrices are built once in the parent and shared
  copy-on-write, so the cores are used without duplicating several GB per worker.

MEMORY (this is a 16GB-class machine, and the pipeline is expected to respect that):
the union of three top-k lists is accumulated in flat numpy arrays and deduplicated
with ``np.unique``, never in a Python dict. A dict keyed by ``(query, ref)`` tuples
costs >200 bytes per entry; at 20k queries x 460 retrieved that is ~2GB *per worker*,
which is what an earlier revision of this file died of. The array form is ~17 bytes per
entry and the query chunk is smaller, so a worker peaks in the low hundreds of MB.
Reranking is additionally sub-chunked, because materialising the candidate strings as
Python lists is itself a large transient.

Output per (split, country):
    <work>/s1/<split>_<country>_cand.npz
        s1_idx, ref_idx            int32   indices into the stage-0 shards
        cos_blob, cos_addr, cos_name  float32  retrieval scores (0 if not retrieved
                                   by that index)
        sample_idx                 int64   which S1 rows were blocked (train only)
"""

import argparse
import os

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import HashingVectorizer

from common import CONFIG, log, rss_gb, work_path

N_FEATURES = 2 ** 21          # hashed n-gram space
MAX_NGRAM_DF_FRAC = 0.01      # drop n-grams occurring in more than this share of refs
REF_BLOCK = 1_000_000         # reference rows hashed at once; bounds peak memory
QUERY_CHUNK = 4_000           # queries per worker task (see MEMORY note above)
RERANK_CHUNK = 400_000        # candidate pairs whose strings are materialised at once
IDF_SAMPLE = 600_000          # docs sampled to estimate document frequency

# Retrieval depth per index, and how many survive the rerank.
K_BLOB, K_ADDR, K_NAME = 200, 200, 60
KEEP = 60

# Set by the parent before forking; workers read these copy-on-write.
_G = {}


def _hasher():
    return HashingVectorizer(
        analyzer="char_wb", ngram_range=CONFIG["ngram_range"], n_features=N_FEATURES,
        norm=None, alternate_sign=False, dtype=np.float32,
    )


def load_shard(split, country, kind, columns=None):
    """Load a stage-0 shard, keeping text columns as object arrays of Python str.

    Deliberately *not* ``.astype(str)``: that produces a fixed-width numpy unicode dtype
    sized to the longest value, which for a 5M-row shard of 200-character addresses
    costs several GB. Object arrays keep one pointer per row.
    """
    path = work_path("s0", f"{split}_{country}_{kind}.parquet")
    if not os.path.exists(path):
        return None
    t = pq.read_table(path, columns=columns)
    return {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}


def blob(shard):
    return [f"{n} {a}" for n, a in zip(shard["name_norm"], shard["addr_norm"])]


def apply_idf(counts, idf):
    """Sublinear TF, IDF weighting, drop pruned n-grams, L2-normalize rows.

    Features with IDF exactly 0 were pruned as too common; ``eliminate_zeros`` physically
    removes them, which is what shrinks the sparse product rather than merely
    down-weighting it.
    """
    m = counts.tocsr(copy=False)
    np.log1p(m.data, out=m.data)
    m.data *= idf[m.indices]
    m.eliminate_zeros()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    m.data /= np.repeat(norms, np.diff(m.indptr))
    return m


def compute_idf(texts, hv, seed=0):
    """IDF vector over the reference set, with very common n-grams pruned to zero.

    Document frequencies are estimated from a sample: the cutoff is a *fraction* of
    documents, so exact counts are unnecessary and sampling saves a full hashing pass
    over millions of records.
    """
    n = len(texts)
    if n > IDF_SAMPLE:
        idx = np.random.default_rng(seed).choice(n, size=IDF_SAMPLE, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for s in range(0, len(sample), REF_BLOCK):
        m = hv.transform(sample[s:s + REF_BLOCK]).tocsr()
        # Indices within a CSR row are unique, so bincount over all of them is exactly
        # the document frequency.
        df += np.bincount(m.indices, minlength=N_FEATURES)
        del m
    idf = (np.log((1.0 + len(sample)) / (1.0 + df)) + 1.0).astype(np.float32)
    idf[df > MAX_NGRAM_DF_FRAC * len(sample)] = 0.0
    del df
    return idf


def build_ref_matrix(texts, hv, idf):
    """Transposed reference matrix (features x records), built in blocks.

    Blocks are released as soon as they are stacked, and the intermediate CSR is
    dropped before returning: the transpose already holds a full copy, so keeping both
    alive doubles a multi-GB matrix for no reason.
    """
    import scipy.sparse as sp
    parts = []
    for s in range(0, len(texts), REF_BLOCK):
        parts.append(apply_idf(hv.transform(texts[s:s + REF_BLOCK]), idf))
    if len(parts) == 1:
        R = parts[0]
        del parts
        return R.T.tocsr()
    R = sp.vstack(parts, format="csr")
    parts.clear()
    del parts
    RT = R.T.tocsr()
    del R
    return RT


def _topk_csr(S, k):
    """Top-k entries of every row of a CSR similarity matrix."""
    rows, cols, vals = [], [], []
    indptr, indices, data = S.indptr, S.indices, S.data
    for i in range(S.shape[0]):
        lo, hi = indptr[i], indptr[i + 1]
        n = hi - lo
        if n == 0:
            continue
        d, ix = data[lo:hi], indices[lo:hi]
        if n > k:
            sel = np.argpartition(d, n - k)[n - k:]
            d, ix = d[sel], ix[sel]
        rows.append(np.full(len(ix), i, dtype=np.int32))
        cols.append(ix.astype(np.int32))
        vals.append(d.astype(np.float32))
    if not rows:
        e = np.empty(0, np.int32)
        return e, e, np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _empty_result():
    e32, ef = np.empty(0, np.int32), np.empty(0, np.float32)
    return e32, e32, ef, ef, ef


def _retrieve(start, stop):
    """Union the three indexes' top-k lists for one query chunk.

    Returns (local_query_idx, ref_idx, scores[n,3]) with local_query_idx sorted
    ascending. The union is done by encoding each (query, ref) pair as a single int64
    and calling ``np.unique`` - a vectorised group-by that costs ~17 bytes per retrieved
    hit, against >200 for the dict-of-tuples this replaced.
    """
    s1 = _G["s1"]
    nref = _G["nref"]
    hv = _hasher()
    specs = [
        ("blob", [f"{n} {a}" for n, a in zip(s1["name_norm"][start:stop],
                                             s1["addr_norm"][start:stop])], K_BLOB, 0),
        ("addr", list(s1["addr_norm"][start:stop]), K_ADDR, 1),
        ("name", list(s1["name_norm"][start:stop]), K_NAME, 2),
    ]
    qs, rs, vs, ss = [], [], [], []
    for key, qtexts, k, slot in specs:
        Q = apply_idf(hv.transform(qtexts), _G[f"idf_{key}"])
        S = (Q @ _G[f"RT_{key}"]).tocsr()
        del Q
        r, c, v = _topk_csr(S, k)
        del S
        if len(r):
            qs.append(r); rs.append(c); vs.append(v)
            ss.append(np.full(len(r), slot, dtype=np.int8))
        del r, c, v
    if not qs:
        return None
    q = np.concatenate(qs); r = np.concatenate(rs)
    v = np.concatenate(vs); sl = np.concatenate(ss)
    qs.clear(); rs.clear(); vs.clear(); ss.clear()

    # Encode (query, ref) into one int64 key. nref <= ~7M and the chunk is 4k rows, so
    # the product is far inside int64 range.
    key = q.astype(np.int64) * nref + r
    del q, r
    uniq, inv = np.unique(key, return_inverse=True)
    del key
    sc = np.zeros((len(uniq), 3), dtype=np.float32)
    # A (query, ref) pair can appear at most once per index, so no accumulation is
    # needed - a plain scatter is correct here.
    sc[inv, sl] = v
    del inv, v, sl
    keys = (uniq // nref).astype(np.int32)   # already sorted: np.unique sorts
    refs = (uniq % nref).astype(np.int32)
    del uniq
    return keys, refs, sc


def _process_chunk(args):
    """Retrieve + rerank one chunk of Source-1 rows. Runs in a forked worker."""
    start, stop = args
    got = _retrieve(start, stop)
    if got is None:
        return _empty_result()
    keys, refs, sc = got
    s1, ref = _G["s1"], _G["ref"]

    # --- rerank: cut the pool to the candidates the model will actually score --------
    # Sub-chunked: the .tolist() calls below turn candidate pairs into Python string
    # lists, which is the single largest transient in this function.
    qi = keys.astype(np.int64) + start
    n_total = len(keys)
    score = np.empty(n_total, dtype=np.float32)
    P = lambda a, b, s: process.cpdist(a, b, scorer=s, workers=1, dtype=np.float32) / 100.0
    for s0 in range(0, n_total, RERANK_CHUNK):
        s1e = min(s0 + RERANK_CHUNK, n_total)
        qq, rr = qi[s0:s1e], refs[s0:s1e]
        an = s1["name_norm"][qq].tolist(); bn = ref["name_norm"][rr].tolist()
        aa = s1["addr_norm"][qq].tolist(); ba = ref["addr_norm"][rr].tolist()
        ak = s1["skel"][qq].tolist(); bk = ref["skel"][rr].tolist()
        nmax = np.maximum.reduce([P(an, bn, fuzz.token_set_ratio), P(an, bn, fuzz.ratio),
                                  P(an, bn, fuzz.token_sort_ratio), P(ak, bk, fuzz.ratio)])
        amax = np.maximum(P(aa, ba, fuzz.token_set_ratio), P(aa, ba, fuzz.ratio))
        del an, bn, aa, ba, ak, bk
        score[s0:s1e] = nmax + amax + sc[s0:s1e, 0] + sc[s0:s1e, 1] + sc[s0:s1e, 2]
        del nmax, amax

    n_local = stop - start
    bounds = np.searchsorted(keys, np.arange(n_local + 1))
    sel = []
    for i in range(n_local):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo <= 0:
            continue
        if hi - lo <= KEEP:
            sel.append(np.arange(lo, hi))
        else:
            sel.append(lo + np.argpartition(-score[lo:hi], KEEP - 1)[:KEEP])
    if not sel:
        return _empty_result()
    sel = np.concatenate(sel)
    sel.sort()
    return (qi[sel].astype(np.int32), refs[sel],
            sc[sel, 0].copy(), sc[sel, 1].copy(), sc[sel, 2].copy())


def block_country(split, country, sample_idx=None, workers=4, max_ref=0):
    s1 = load_shard(split, country, "s1")
    ref = load_shard(split, country, "ref")
    if s1 is None or ref is None:
        log(f"  {split}/{country}: missing shard, skipped")
        return
    if max_ref and max_ref < len(ref["entity_id"]):
        # Smoke-test escape hatch: shrink the reference pool so a full run of every
        # stage can be exercised cheaply. It must be a *prefix*, never a random
        # sample - the emitted ref_idx values index the full stage-0 shard that later
        # stages reload, and any other subset would silently misalign them.
        ref = {k: v[:max_ref] for k, v in ref.items()}
        log(f"  {split}/{country}: reference pool capped at {max_ref:,} (smoke test)")
    if sample_idx is not None:
        # Training blocks only a sample of Source-1 entities - a ~30-feature GBDT does
        # not need 2.2M of them - but always against the FULL reference pool, so the
        # candidate distribution matches inference.
        s1 = {k: v[sample_idx] for k, v in s1.items()}
    n1, nref = len(s1["entity_id"]), len(ref["entity_id"])
    log(f"  {split}/{country}: S1={n1:,} ref={nref:,}  [rss {rss_gb():.2f}GB]")

    hv = _hasher()
    _G.clear()
    _G["s1"], _G["ref"], _G["nref"] = s1, ref, nref
    for key in ("blob", "addr", "name"):
        if key == "blob":
            texts = blob(ref)
        elif key == "addr":
            texts = list(ref["addr_norm"])
        else:
            texts = list(ref["name_norm"])
        idf = compute_idf(texts, hv)
        _G[f"idf_{key}"] = idf
        _G[f"RT_{key}"] = build_ref_matrix(texts, hv, idf)
        del texts
        log(f"  {split}/{country}: index '{key}' built "
            f"({_G[f'RT_{key}'].nnz/max(nref,1):.1f} nnz/record) [rss {rss_gb():.2f}GB]")

    tasks = [(s, min(s + QUERY_CHUNK, n1)) for s in range(0, n1, QUERY_CHUNK)]
    results = []
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        # fork (not spawn) so the several-GB reference matrices are shared
        # copy-on-write instead of pickled to each worker.
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for i, out in enumerate(pool.imap(_process_chunk, tasks, chunksize=1)):
                results.append(out)
                if i % 10 == 0 or i + 1 == len(tasks):
                    log(f"    chunk {i+1}/{len(tasks)} [rss {rss_gb():.2f}GB]")
    else:
        for i, t in enumerate(tasks):
            results.append(_process_chunk(t))
            if i % 10 == 0 or i + 1 == len(tasks):
                log(f"    chunk {i+1}/{len(tasks)} [rss {rss_gb():.2f}GB]")

    s1_idx = np.concatenate([r[0] for r in results])
    ref_idx = np.concatenate([r[1] for r in results])
    cb = np.concatenate([r[2] for r in results])
    ca = np.concatenate([r[3] for r in results])
    cn = np.concatenate([r[4] for r in results])
    results.clear()
    del results

    out = work_path("s1", f"{split}_{country}_cand.npz")
    np.savez(out, s1_idx=s1_idx, ref_idx=ref_idx, cos_blob=cb, cos_addr=ca, cos_name=cn,
             sample_idx=(sample_idx if sample_idx is not None else np.empty(0, np.int64)))
    log(f"  {split}/{country}: {len(s1_idx):,} candidates "
        f"({len(s1_idx)/max(n1,1):.1f} per S1) -> {os.path.basename(out)}")
    _G.clear()


def auto_workers():
    """Pick a worker count from the cores and RAM actually available.

    Workers are forked and read the reference matrices copy-on-write, so each one adds
    only its own transients (a few hundred MB) rather than a copy of the index. The
    binding constraints are therefore (a) cores, and (b) leaving the *parent* enough
    room for the largest index plus its shard - about 11GB on the biggest country here.

    Past ~16 workers the sparse products contend for memory bandwidth rather than
    going faster, so that is the cap regardless of core count.
    """
    cores = os.cpu_count() or 4
    n = max(1, min(16, cores - 2))
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, AttributeError, OSError):
        ram_gb = None
    if ram_gb:
        # Reserve ~12GB for the parent's index + shard, then ~0.6GB per worker.
        budget = max(1, int((ram_gb - 12) / 0.6))
        n = max(1, min(n, budget))
        log(f"  auto workers={n} ({cores} cores, {ram_gb:.0f}GB RAM)")
    else:
        log(f"  auto workers={n} ({cores} cores, RAM unknown)")
    return n


def countries(split):
    d = os.path.dirname(work_path("s0", "x"))
    return sorted({fn[len(split) + 1:-len("_s1.parquet")]
                   for fn in os.listdir(d)
                   if fn.startswith(split + "_") and fn.endswith("_s1.parquet")})


def sample_for_train(split, cs, n_total, seed):
    """Deterministic per-country Source-1 sample, proportional to country size."""
    rng = np.random.default_rng(seed)
    sizes = {c: pq.read_metadata(work_path("s0", f"{split}_{c}_s1.parquet")).num_rows
             for c in cs}
    total = sum(sizes.values())
    return {c: np.sort(rng.choice(sizes[c],
                                  size=min(sizes[c], max(1, int(round(n_total * sizes[c] / total)))),
                                  replace=False)).astype(np.int64)
            for c in cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto: leave 2 cores for the parent, cap at 8")
    ap.add_argument("--countries", default="", help="comma-separated subset, for smoke runs")
    ap.add_argument("--train-sample", type=int, default=None,
                    help="override total Source-1 training entities to block")
    ap.add_argument("--max-ref", type=int, default=0,
                    help="cap the reference pool (prefix) - smoke tests only")
    args = ap.parse_args()
    if args.workers <= 0:
        args.workers = auto_workers()
    only = {c for c in args.countries.split(",") if c}
    for split in args.splits:
        cs = [c for c in countries(split) if not only or c in only]
        log(f"=== blocking {split}: countries={cs} ===")
        samples = None
        if split == "train":
            n = (args.train_sample if args.train_sample is not None
                 else CONFIG["train_sample"] + CONFIG["holdout_sample"])
            samples = sample_for_train(split, cs, n, CONFIG["seed"])
            log(f"  training sample: { {c: len(v) for c, v in samples.items()} }")
        for c in cs:
            block_country(split, c, None if samples is None else samples[c],
                          args.workers, args.max_ref)
    log("stage 1 complete")


if __name__ == "__main__":
    main()
