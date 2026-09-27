"""
Blocking quality report: recall ceiling and reduction ratio on the training split.

Recall here is an upper bound on what any downstream model can achieve - a true match
that blocking never proposes can never be predicted. This runs before the model is
trained so a bad blocking configuration is visible immediately rather than after hours
of training.

Two numbers are reported:

* pair recall     - fraction of ground-truth links present in the candidate set.
* entity recall   - fraction of Source-1 entities whose true matches are *all* present.
  This is the one that matters for the macro-averaged F_0.5 metric, which scores per
  entity rather than per link.

Reduction ratio is reported alongside, since a blocker can always buy recall by
proposing more pairs and the trade-off has to be visible.
"""

import os

import numpy as np
import pyarrow.parquet as pq

from common import log, read_ground_truth, work_path
from s1_block import countries


def main():
    gt = read_ground_truth()
    split = "train"
    tot_links = tot_hit = 0
    tot_ent = full_ent = single_ent = 0
    tot_pairs = tot_space = 0

    for country in countries(split):
        cand_path = work_path("s1", f"{split}_{country}_cand.npz")
        if not os.path.exists(cand_path):
            log(f"{country}: no candidate file, skipped")
            continue
        z = np.load(cand_path, allow_pickle=False)
        s1_idx, ref_idx = z["s1_idx"], z["ref_idx"]
        sample_idx = z["sample_idx"]

        s1_ids = pq.read_table(work_path("s0", f"{split}_{country}_s1.parquet"),
                               columns=["entity_id"]).column(0).to_numpy(zero_copy_only=False)
        ref_ids = pq.read_table(work_path("s0", f"{split}_{country}_ref.parquet"),
                                columns=["entity_id"]).column(0).to_numpy(zero_copy_only=False)
        if len(sample_idx):
            s1_ids = s1_ids[sample_idx]

        n1 = len(s1_ids)
        tot_pairs += len(s1_idx)
        tot_space += n1 * len(ref_ids)

        # Group candidates by Source-1 row (s1_idx is sorted by construction).
        bounds = np.searchsorted(s1_idx, np.arange(n1 + 1))
        c_links = c_hit = 0
        c_ent = c_full = c_single = 0
        for i in range(n1):
            truth = gt.get(str(s1_ids[i]))
            if truth is None:
                continue
            c_ent += 1
            if not truth:
                c_single += 1
                c_full += 1  # nothing to miss
                continue
            cands = set(ref_ids[ref_idx[bounds[i]:bounds[i + 1]]].tolist())
            hits = sum(1 for t in truth if t in cands)
            c_links += len(truth)
            c_hit += hits
            if hits == len(truth):
                c_full += 1
        log(f"{country}: entities={c_ent:,} pair_recall={100*c_hit/max(c_links,1):.2f}% "
            f"entity_recall(all matches found)={100*c_full/max(c_ent,1):.2f}% "
            f"singletons={c_single:,} cands/S1={len(s1_idx)/max(n1,1):.1f}")
        tot_links += c_links
        tot_hit += c_hit
        tot_ent += c_ent
        full_ent += c_full
        single_ent += c_single

    log("=" * 70)
    log(f"OVERALL pair recall   : {100*tot_hit/max(tot_links,1):.3f}%  "
        f"({tot_hit:,}/{tot_links:,} links)")
    log(f"OVERALL entity recall : {100*full_ent/max(tot_ent,1):.3f}%  "
        f"({full_ent:,}/{tot_ent:,} entities have every true match in the candidate set)")
    log(f"candidate pairs       : {tot_pairs:,}")
    log(f"reduction ratio       : {1 - tot_pairs/max(tot_space,1):.8f} "
        f"(full space was {tot_space:,} pairs)")


if __name__ == "__main__":
    main()
