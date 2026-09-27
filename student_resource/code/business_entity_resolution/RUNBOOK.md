# Runbook — exact commands, timings, memory

All timings and memory figures below are **measured on this machine** (M4, 10 cores,
16GB RAM) or extrapolated from measured throughput, not guessed. Extrapolations are
marked *(est.)*.

Throughput measured at 3 workers: **~84 queries/sec** against an India-density pool
(4.13M refs, 91.1 nnz/record). US and France are lighter (~47 nnz/record) and run
roughly 2x faster per query.

> Running on a hosted notebook (SageMaker, Colab)? Use **`run_pipeline.ipynb`** at the
> repository root instead — it wraps every command below with live output and an
> up-front instance-size check. Note that a SageMaker notebook instance defaults to a
> **5GB EBS volume**, which is far too small; request at least 50GB at creation time.

```bash
cd <repo root>
PY=.venv/bin/python
cd student_resource/code/business_entity_resolution/src
```

Paths are overridable with `BER_DATA_DIR`, `BER_OUT_DIR` and `BER_WORK_DIR`.

## 0. Clear the smoke-test artifacts first

**Required.** The candidate/feature files currently in `work/` were produced with a
deliberately capped reference pool (`--max-ref`) and are not valid. Reusing them
silently produces a garbage submission.

```bash
rm -rf ../../../../work/s1 ../../../../work/s2 ../../../../work/s3
rm -f ../../../../student_resource/output/*.tsv
```

Keep `work/s0` — stage 0 already ran correctly over the real data and takes ~5 min to
redo if you delete it.

## 1. Blocking (the long pole, ~5h est.)

Run each country separately so a failure costs one country, not the whole stage. Each
command is independently resumable — rerunning one overwrites only its own `.npz`.

```bash
$PY s1_block.py train --countries India  --workers 3     #  ~25 min (est.)
$PY s1_block.py train --countries US     --workers 3     #  ~35 min (est.)
$PY s1_block.py test  --countries France --workers 4     #  ~20 min (est.)
$PY s1_block.py test  --countries US     --workers 3     #  ~1.2 h  (est.)
$PY s1_block.py test  --countries India  --workers 2     #  ~2.7 h  (est.)
```

`--workers 2` on test/India is deliberate. That shard has 4.72M references — the
largest index in the job, ~7.5GB resident (est., measured 6.67GB at 4.13M). During the
probe, system swap reached 3.8GB of 4GB at 3 workers on the *smaller* train/India pool.
Two workers trades ~30 min for headroom. If you are watching the run and swap stays
low, 3 is fine.

Every stage prints `[rss N.NNGB]`. If that number climbs past ~10GB, stop and drop
`--workers`.

## 2. Blocking quality check (~1 min, optional but recommended)

```bash
$PY eval_blocking.py
```

Measured on a 5,000-entity India sample against the full reference pool:
**pair recall 91.65%, entity recall 79.40%, reduction ratio 0.99998548.**
If the full-scale number comes out far below ~90%, something regressed in stage 1 —
investigate before spending hours on stages 2-4.

## 3. Features (~15 min est., ~5GB peak)

```bash
$PY s2_features.py train
$PY s2_features.py test
```

Writes ~12.5GB of memmapped float32 to `work/s2/`. Peak RSS stays near 5GB because the
matrix is never resident — measured 4.86GB on a 15.5M-pair shard.

## 4. Train + tune the decision rule (~20 min est.)

```bash
$PY s3_train.py
```

Prints the swept `(threshold, margin)` grid with the macro-F_0.5 each pair achieves on
an entity-disjoint holdout, then writes `work/s3/model.txt` and `work/s3/decision.json`.
**The `BEST: macro F0.5 = ...` line is your validation score** — that is the number to
put in the methodology document, and your best available predictor of the leaderboard.

Tuning knobs if memory is tight: `--neg-frac 0.15` (fewer negatives kept for the fit)
and `--holdout-frac 0.15`.

## 5. Predict and write the submission (~10 min est.)

```bash
$PY s4_predict.py
```

Writes both files to `student_resource/output/`.

## 6. Validate — do this every time, before every upload

```bash
cd ../../..        # -> student_resource/
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

`--check-ids` costs ~16s and 4.3GB and is worth it: it is the only check that catches a
corrupted index-to-entity_id mapping, which would otherwise cost you a whole submission.
Expect `PASS — no blocking issues found.` and exactly **1,732,544** rows in each file.

## Disk

`work/` peaks around **20GB** (s0 1.1GB + s1 ~2.5GB + s2 ~15GB). You had 165GB free.

## If you later want it faster

Two untested levers, in order of expected payoff. Measure recall with
`eval_blocking.py` after changing either — do not apply them blind:

1. `MAX_NGRAM_DF_FRAC` in `s1_block.py`: `0.01` -> `0.002`. Prunes far more common
   n-grams, which shrinks every posting list. This is the single biggest speed and
   memory lever; the module docstring reports 323,109 -> 1,416 non-zero similarities
   per query from pruning at this order of magnitude.
2. `K_BLOB, K_ADDR` : `200, 200` -> `120, 120`. Cuts top-k selection and rerank cost
   roughly 40%.

Re-run `s1_block.py train --countries India --train-sample 5000` then
`eval_blocking.py`, and compare against the 91.65% baseline above.
