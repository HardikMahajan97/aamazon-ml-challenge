# Business Entity Resolution — pipeline

Matches each Source-1 business record to its Source-2 / Source-3 duplicates across
three noisy, independently-sourced feeds. Classical retrieve → rerank → classify
entity resolution: character n-gram TF-IDF blocking, ~30 hand-built pairwise
similarity features, and a LightGBM binary classifier whose decision rule is tuned
directly against the competition metric.

No external data, no lookups, no pretrained weights. `lightgbm` (MIT) is the only
model, far below the 8B parameter cap.

## Reproducing end-to-end

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd src

python s0_prepare.py                    # normalize + shard by country   (~5 min)
python s1_block.py train test           # candidate generation           (hours; see below)
python eval_blocking.py                 # blocking recall ceiling        (optional)
python s2_features.py train test        # pairwise features
python s3_train.py                      # fit model + tune decision rule
python s4_predict.py                    # write the two output TSVs
```

Outputs land in `student_resource/output/`:

- `matching_results.tsv` — final matches (the leaderboard file)
- `candidate_pairs.tsv` — the candidate set fed to the model

Validate before submitting, from `student_resource/`:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

Large intermediates go to `$BER_WORK_DIR` (default `<repo>/work`), deliberately
outside the submission tree. Budget ~25GB.

## Stages

| Stage | File | What it does |
| --- | --- | --- |
| 0 | `s0_prepare.py` | Streams the 2.3GB of TSV once. Folds accents, romanizes Indic scripts, canonicalizes legal suffixes / street types / state names, and shards by country to parquet. |
| 1 | `s1_block.py` | Three char-n-gram TF-IDF indexes (name, address, name+address) per country. Unions their top-k lists, reranks by fuzzy similarity, keeps the top 60 per entity. |
| 2 | `s2_features.py` | ~30 float32 features per candidate pair, written through a memmap. |
| 3 | `s3_train.py` | LightGBM binary classifier; sweeps (threshold, margin) against the real macro-F_0.5 on an entity-disjoint holdout. |
| 4 | `s4_predict.py` | Scores test candidates, applies the tuned rule, writes both TSVs. |

`common.py` holds the shared normalization — every stage imports it, so training and
test records are guaranteed to be normalized identically.

## Why the metric shapes the design

F_0.5 is macro-averaged **per Source-1 entity**, and an entity with no true matches
scores a full 1.0 for a correctly empty prediction. Two consequences drive the code:

1. **Precision dominates.** Beyond a probability threshold, stage 3 also applies a
   *margin* rule: a candidate must be within `margin` of the entity's best probability.
   This suppresses the long tail of similarly-named businesses on the same street,
   which is the dominant false-merge mode in this data.
2. **Singletons are worth chasing.** They are a large share of the test set, and the
   tuning loop scores them exactly as the leaderboard does, so the sweep will happily
   choose a conservative threshold to bank them.

## Country handling

`country` is treated as an **open set of string labels**, never an enum. Stage 0
slugifies whatever labels it finds and shards accordingly; every later stage discovers
countries from the shard filenames. France appears only at test time and flows through
with no special-casing. Records are never compared across countries — country agrees
on 100% of training ground-truth links, so this is a sound hard partition rather than
an approximation.

## Memory

Written for a 16GB machine. The constraints that matter:

- Text columns stay as object arrays of Python `str`. `.astype(str)` would produce a
  fixed-width unicode dtype sized to the longest value — several GB per shard.
- Stage 1 accumulates the union of three top-k lists in flat numpy arrays keyed by a
  packed `int64`, never a Python dict. A dict of `(query, ref)` tuples costs >200 bytes
  per entry and will exhaust RAM at this scale.
- Stage 2 writes features through `np.lib.format.open_memmap`, a chunk at a time; the
  full test matrix is ~11GB and is never resident.
- Reference matrices are built once in the parent and shared with forked workers
  copy-on-write, so worker count does not multiply the index cost.

Every stage logs peak RSS. Tune `--workers` (stage 1) down first if memory is tight.
