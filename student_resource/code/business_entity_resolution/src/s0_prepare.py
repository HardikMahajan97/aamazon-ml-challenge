"""
Stage 0 - normalize every source file once and shard it by country.

This is the only full pass over the raw 2.3GB of TSV. Everything downstream reads the
compact per-country shards this writes, so the expensive normalization (accent folding,
romanization, token canonicalization) is paid exactly once.

Sharding by country is safe and not an approximation: country agrees on 100% of the
7.6M ground-truth links, so two records with different country labels can never match.
For the reference sources (S2, S3) we write one shard per country holding both sources
together, because candidate generation searches their union.

Output per (split, country):
    <work>/s0/<split>_<country>_s1.parquet     Source-1 records
    <work>/s0/<split>_<country>_ref.parquet    Source-2 + Source-3 records
Columns: entity_id, name_norm, addr_norm, skel, digits, src (2 or 3)
"""

import os
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from common import CONFIG, SOURCE_FILES, data_path, log, normalize_record, stream_tsv, work_path

SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("name_norm", pa.string()),
    ("addr_norm", pa.string()),
    ("skel", pa.string()),
    ("digits", pa.string()),
    ("src", pa.int8()),
])


class ShardWriter:
    """Buffered per-country parquet writer.

    Rows arrive interleaved by country, so each country gets a small in-memory buffer
    that is flushed as a row group once it fills. This keeps peak memory at
    (n_countries x buffer) instead of holding a whole 5M-row file.
    """

    def __init__(self, path_fn, buffer_rows=250_000):
        self.path_fn = path_fn
        self.buffer_rows = buffer_rows
        self.buffers = defaultdict(list)
        self.writers = {}
        self.counts = defaultdict(int)

    def add(self, country, row):
        buf = self.buffers[country]
        buf.append(row)
        self.counts[country] += 1
        if len(buf) >= self.buffer_rows:
            self._flush(country)

    def _flush(self, country):
        buf = self.buffers[country]
        if not buf:
            return
        cols = list(zip(*buf))
        table = pa.Table.from_arrays(
            [
                pa.array(cols[0], pa.string()),
                pa.array(cols[1], pa.string()),
                pa.array(cols[2], pa.string()),
                pa.array(cols[3], pa.string()),
                pa.array(cols[4], pa.string()),
                pa.array(cols[5], pa.int8()),
            ],
            schema=SCHEMA,
        )
        if country not in self.writers:
            self.writers[country] = pq.ParquetWriter(self.path_fn(country), SCHEMA, compression="zstd")
        self.writers[country].write_table(table)
        buf.clear()

    def close(self):
        for country in list(self.buffers):
            self._flush(country)
        for w in self.writers.values():
            w.close()


def prepare_split(split):
    """Normalize the three source files of one split into per-country shards."""
    s1_writer = ShardWriter(lambda c: work_path("s0", f"{split}_{_slug(c)}_s1.parquet"))
    ref_writer = ShardWriter(lambda c: work_path("s0", f"{split}_{_slug(c)}_ref.parquet"))

    for src in (1, 2, 3):
        path = data_path(SOURCE_FILES[(split, src)])
        writer = s1_writer if src == 1 else ref_writer
        n = 0
        for row in stream_tsv(path):
            # entity_id, business_name, business_address, country
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            entity_id, name, addr, country = row[0], row[1], row[2], row[3]
            name_norm, addr_norm, skel, digits = normalize_record(name, addr)
            writer.add(country, (entity_id, name_norm, addr_norm, skel, digits, src))
            n += 1
            if n % 1_000_000 == 0:
                log(f"  {split} source{src}: {n:,} rows")
        log(f"  {split} source{src}: {n:,} rows (done)")

    s1_writer.close()
    ref_writer.close()
    log(f"{split} S1 per country:  {dict(s1_writer.counts)}")
    log(f"{split} ref per country: {dict(ref_writer.counts)}")
    return dict(s1_writer.counts)


def _slug(country):
    """Filesystem-safe country token. Country is an open set of labels (test adds
    France, and more could appear), so it is slugified rather than mapped to a fixed
    enum."""
    return "".join(ch if ch.isalnum() else "_" for ch in country) or "unknown"


def countries_for(split):
    """Countries present in a prepared split, discovered from the shard filenames."""
    d = work_path("s0", "x")[:-1]
    out = []
    for fn in sorted(os.listdir(os.path.dirname(d))):
        prefix, suffix = f"{split}_", "_s1.parquet"
        if fn.startswith(prefix) and fn.endswith(suffix):
            out.append(fn[len(prefix):-len(suffix)])
    return out


def main():
    os.makedirs(CONFIG["work_dir"], exist_ok=True)
    for split in ("train", "test"):
        log(f"=== preparing {split} ===")
        prepare_split(split)
    log("stage 0 complete")


if __name__ == "__main__":
    main()
