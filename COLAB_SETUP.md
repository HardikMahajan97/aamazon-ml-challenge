# Running on Google Colab — setup steps

Target runtime: **24 cores / 47 GB RAM / ~200 GB disk** (verified). The TPU itself is
unused — this pipeline is CPU and memory bound — but that runtime's host VM is a good
fit. A standard high-RAM CPU runtime works equally well.

Expected total runtime: **~2–3 hours.**

---

## Step 1 — Package the dataset locally (one time)

Colab cannot pull the dataset from GitHub: it is 2.3 GB and every source file exceeds
GitHub's 100 MB limit. Upload it to Drive instead — as a single compressed archive,
because seven separate multi-hundred-MB browser uploads is a reliability problem.

Run this on your Mac, from the repository root:

```bash
cd student_resource
tar czf ~/Desktop/dataset.tar.gz dataset/
ls -lh ~/Desktop/dataset.tar.gz
```

The TSVs compress about 2.3x, so expect **~1.0 GB**. Takes a few minutes.

## Step 2 — Upload to Drive

In Google Drive, create a folder named `Amazon_ML_Challeneg` at the top level of *My
Drive*, and upload `dataset.tar.gz` into it. Final path:

```
My Drive/Amazon_ML_Challeneg/dataset.tar.gz
```

Use the desktop Drive app or drag-and-drop in the browser. At ~1 GB this is the slowest
step; it is also the only one you ever repeat.

If you keep the archive somewhere else, edit the `ARCHIVE` path in notebook section 4.

## Step 3 — Push the code to GitHub

The code is tiny (400 KB) and already git-ready. From the repository root:

```bash
git add .
git commit -m "Business entity resolution pipeline"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`.gitignore` already excludes the dataset, `work/`, `.venv/` and the outputs, so this
pushes only source and docs.

## Step 4 — Open Colab and clone

New notebook → **Runtime → Change runtime type** → pick the runtime you measured
(24 cores / 47 GB). Then in the first cell:

```python
!git clone https://github.com/<you>/<repo>.git /content/Amazon_ML_Challeneg
```

Then **File → Open notebook → GitHub**, point at your repo, and open
`run_pipeline.ipynb`. Alternatively upload the `.ipynb` directly.

> If the repo is private, clone with a personal access token:
> `!git clone https://<token>@github.com/<you>/<repo>.git /content/Amazon_ML_Challeneg`

## Step 5 — Run the notebook top to bottom

The notebook detects Colab automatically and sets:

| | |
|---|---|
| `REPO` | `/content/Amazon_ML_Challeneg` |
| `DATA_DIR` | `/content/dataset` (extracted from Drive in section 4) |
| `WORK_DIR` | `/content/work` — local disk, **not** Drive |
| workers | auto → **16** on this runtime |

`work/` must stay on local disk. It does ~20 GB of random I/O and Drive's FUSE mount
would dominate the runtime.

Two checkpoints worth respecting:

- **Section 1** fails loudly if the runtime is too small. Do not skip it.
- **Section 8** (`eval_blocking`) should report pair recall near **91.65%**. If it does
  not, stop — do not spend hours on the later stages over a broken candidate set.

## Step 6 — Survive disconnects

This is the real risk on Colab, not resources. A 2–3 hour run gives the session plenty
of opportunity to drop.

The pipeline is built for this: every stage checkpoints to disk, and stage 1 — the long
one — is **one cell per country**. After each country cell, run:

```python
sync_checkpoints()      # copies work/s0, work/s1, work/s3 to Drive
```

If the session dies, re-clone, re-run sections 1–5, copy the checkpoints back, and
resume from whichever country was in flight:

```python
!mkdir -p /content/work && cp -ru /content/drive/MyDrive/Amazon_ML_Challeneg/work/* /content/work/
```

Nothing already completed is recomputed.

## Step 7 — Get the results out

`/content` is ephemeral. Before the session ends, copy the outputs to Drive:

```python
!cp /content/Amazon_ML_Challeneg/student_resource/output/*.tsv \
    /content/drive/MyDrive/Amazon_ML_Challeneg/
```

`matching_results.tsv` is what you upload to the leaderboard. The final notebook
section builds the full submission zip.

---

## Notes and gotchas

**Dependencies.** The notebook does *not* apply `requirements.txt` on Colab. Colab
already ships numpy, scipy, scikit-learn, pandas and pyarrow, and forcing the pinned
versions triggers a large reinstall plus a mandatory kernel restart for no benefit.
Only `anyascii`, `rapidfuzz` and `lightgbm` are installed. Section 2 then verifies the
imports and asserts that `rapidfuzz.process.cpdist` exists — it is the one API the
feature stages depend on that an older rapidfuzz would lack.

**Disk.** Peak usage is roughly 2.3 GB (dataset) + 1 GB (archive) + ~20 GB (`work/`)
≈ 24 GB, against ~200 GB available.

**`--neg-frac`.** Stage 3 auto-selects `1.0` on hosts above 30 GB RAM, training on
every negative instead of a quarter of them. Better calibration means a better tuned
threshold.

**Optional recall lever.** `KEEP = 60` in `s1_block.py` yields a measured 91.65% pair
recall against a 94.33% retrieval ceiling. With 47 GB you can afford `KEEP = 80` to
recover part of that gap, at ~40% more candidate pairs. Confirm you reproduce 91.65% at
60 first, then change one thing at a time. This is untested.

**What has not been verified.** The notebook has never been executed, on Colab or
anywhere else. The individual stages have all been run successfully and the output
format passes the official validator with `--check-ids`, but the first Colab run is the
first real test of the notebook itself.
