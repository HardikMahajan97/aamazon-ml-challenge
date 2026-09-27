# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> Numbers marked **[FILL]** come from the `s3_train.py` run on the full data and are the
> only gaps left in this document. Everything else is measured and final.

---

## 1. Executive Summary

A three-stage classical entity-resolution pipeline: country-partitioned character
n-gram TF-IDF retrieval, a fuzzy-similarity rerank, and a LightGBM classifier over ~30
pairwise features whose decision rule is tuned directly against macro-F_0.5 rather than
against a generic classification loss. Blocking reduces 20.7 billion possible pairs to
60 candidates per entity while retaining **91.65% of true links**. The main design
commitment is that the metric, not the classifier, is the object of optimisation: a
margin rule on top of the probability threshold is what converts a good ranker into a
precision-heavy matcher.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings that changed the design:

- **`country` is a perfect partition.** It agrees on 100% of training ground-truth
  links, so cross-country comparison is not merely unhelpful, it is provably wasted
  work. Sharding by country reduces the candidate space by more than an order of
  magnitude at zero recall cost. It is treated as an **open set of labels**, never an
  enum — France appears only at test time and flows through unmodified.
- **~7% of Source-2/3 names are in an Indic script** while the Source-1 name is Latin.
  Romanization systematically drops vowels (`कर्नाटक` -> `krnatk` vs `karnataka`), which
  defeats edit distance. A vowel-stripped *consonant skeleton* (`krntk`) realigns them
  and is the cheapest high-value signal in the feature set.
- **~3.3% of reference records have an empty address**, and many names are generic.
  Either field alone can be the only usable signal, which is why retrieval uses three
  independent indexes rather than one concatenated vector.
- **Ground truth is strictly many-to-one.** Every link points to a *distinct* Source-2/3
  record; no reference record is shared between two Source-1 entities. This licenses
  "competition" features: a candidate strongly claimed by a different entity is less
  likely to belong to this one.
- **Noise is adversarial but structured** — legal-suffix variants (Pvt/Private, Corp/
  Corporation), street abbreviations (Rd/Road), `&` vs `and`, word-order transposition,
  landmark addresses ("Near SBI ATM"), and the literal string `null` used as a
  placeholder inside addresses. All are handled by explicit canonicalisation tables in
  `common.py` rather than left to the model.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (retrieve → rerank → classify)
**Core Innovation:** Two things. (a) Candidate generation is a *union of three*
independent TF-IDF indexes followed by a learned-free rerank, because measurement
showed no single index suffices (name alone 57.87%, address alone 83.96%, blob 90.12%,
union 94.33%). (b) The decision rule is a **joint (threshold, margin) sweep scored with
the exact competition metric**, including singletons, rather than a probability cutoff
chosen by F1 or Youden's J.

All normalization lives in one module (`common.py`) imported by every stage, so train
and test records are guaranteed to be transformed identically — divergence there is the
classic silent recall killer in ER pipelines.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** hashed character n-grams (`char_wb`, n=4-5, 2^21 buckets)
  with sublinear TF and IDF weighting, over three separate representations per country:
  business name, business address, and a name+address blob. High-document-frequency
  n-grams are pruned to zero and physically eliminated — without this, generic n-grams
  such as `" rue"` or `"ville"` make every query touch almost the whole reference set
  (measured on French references: 323,109 non-zero similarities per query unpruned
  versus 1,416 pruned).
- **Retrieval depth:** top-200 blob, top-200 address, top-60 name, unioned.
- **Rerank:** the ~460-candidate pool is cut to **60** by
  `name_sim + addr_sim + cos_blob + cos_addr + cos_name`, where `name_sim` is the max of
  four fuzzy scorers over the name and its consonant skeleton. Simpler rules are
  measurably worse — a plain `0.5*name + 0.5*address` average retains only 89.7%.
- **Candidate pairs generated:** 60 per Source-1 entity; ~104M for the full test set.
- **Reduction ratio:** 0.99998548 (from 20,666,730,000 possible pairs in-country).

**How true matches were not lost** — measured, not assumed. `eval_blocking.py` scores
the candidate set against ground truth before any model is trained, on a 5,000-entity
India sample against the *full* 4.13M reference pool:

| Metric | Value |
| --- | --- |
| Pair recall (links present in candidate set) | **91.65%** (15,738 / 17,172) |
| Entity recall (entities with *every* true match present) | 79.40% |
| Candidates per Source-1 entity | 60.0 |

91.65% is the hard ceiling on this pipeline's recall. The three-index union was chosen
specifically because any single index caps far lower.

---

## 4. Matching Model

**Features used** (30 total, all float32):

- **Name features:** `ratio`, `token_sort_ratio` (absorbs word-order transposition),
  `token_set_ratio` (absorbs dropped legal suffixes), `partial_ratio` (catches a name
  contained in a longer one). They are kept separate because they disagree in
  informative ways.
- **Skeleton features:** `ratio` and `partial_ratio` over vowel-stripped names — the
  transliteration signal.
- **Address features:** `ratio`, `token_sort_ratio`, `token_set_ratio`, plus a separate
  `digits_tset` over the extracted numeric components. House numbers and PIN/ZIP codes
  survive almost every kind of textual noise here, so they are compared independently
  of the surrounding address text.
- **Blocking context:** the three retrieval cosines kept separate, their max, how many
  indexes retrieved the pair, the candidate's rank within its entity, the entity's best
  cosine, and the margin below it.
- **Competition features:** the best cosine any *other* entity has for this reference
  record, the margin below it, and how many entities retrieved it. These exploit the
  strict many-to-one structure of the ground truth.
- **Shape features:** name/address length ratios and empty-address flags.

**Model type:** LightGBM binary classifier (MIT licence; far below the 8B parameter
cap). Chosen because the features are ~30 dense numeric similarities with strong
monotone structure and inference must cover ~104M pairs — a GBDT scores that in minutes
on CPU, while a transformer re-ranker would cost orders of magnitude more for a metric
dominated by threshold placement rather than subtle semantics.

Negatives outnumber positives ~20:1. The training split downsamples negatives to fit in
memory; **the holdout is never downsampled**, because precision measured against an
artificial negative density would tune the threshold far too permissively.

**Threshold selection method:** joint sweep of `(threshold, margin)` scored with the
real macro-F_0.5 on an **entity-disjoint** holdout (splitting by pair would leak a
business across both sides and make the tuned threshold optimistic). Singletons are
included exactly as the leaderboard scores them. The `margin` rule — a candidate must
be within `margin` of the entity's best probability — is what suppresses the long tail
of similarly-named businesses on the same street, the dominant false-merge mode here.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **[FILL — the `BEST: macro F0.5 = ...` line from `s3_train.py`]**
- **Selected decision rule:** **[FILL — thr / margin from `work/s3/decision.json`]**
- **Blocking recall ceiling:** 91.65% pair recall / 79.40% entity recall (measured).

**Common false positives (wrong merges):** chain businesses and franchises sharing a
name within one city, and distinct businesses at the same commercial address (malls,
office towers). Both produce genuinely high name *and* address similarity, so no pair
feature separates them. The margin rule is the main defence: when several candidates
score alike, only those near the maximum survive.

**Common false negatives (missed matches):** dominated by the 8.35% of links blocking
never proposes. Within that, the recurring cases are records with an empty address and
a short generic name (too little signal for any index), and heavily transliterated
names where romanization diverges enough that even the consonant skeleton misses.

**Where the remaining headroom is:** recall, not precision. Raising `KEEP` above 60 or
deepening retrieval lifts the 91.65% ceiling directly — but under F_0.5 every extra
candidate is also an opportunity for a false merge, so it must be paired with a
re-tuned margin.

---

## 6. Conclusion

A country-partitioned, three-index TF-IDF blocker retains 91.65% of true links at 60
candidates per entity — a 0.99998548 reduction of the comparison space — and a LightGBM
classifier over 30 pairwise similarity features converts that into final matches. The
decision rule is tuned against the exact competition metric rather than a proxy loss,
which matters more than any single modelling choice given how sharply F_0.5 punishes
false merges. The main lesson: every structural claim (country as a partition, the
many-to-one link structure, which index contributes what recall) was cheap to verify
against ground truth and each one paid for itself — the blocking evaluator running
*before* the model is what kept the pipeline honest.
