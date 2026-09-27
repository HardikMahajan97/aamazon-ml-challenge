"""
Shared config, IO and text normalization for the Business Entity Resolution pipeline.

Every stage imports from here so that the *exact same* normalization is applied to
training records and test records. Divergence between the two is the classic way an
entity-resolution pipeline silently loses recall, so normalization lives in one place.

Design notes that are specific to this dataset (all measured, see the methodology doc):

* ``country`` agrees on 100% of ground-truth links, so it is a hard partition key.
  Every stage shards by country and never compares across countries.
* ~7% of Source-2/3 business names are written in an Indic script while the Source-1
  name is Latin. ``anyascii`` romanizes them into something fuzzy-matchable, but it is
  pure Python, so it is only invoked on strings that actually contain a non-ASCII
  character.
* Indic romanization drops vowels (``कर्नाटक`` -> ``krnatk`` vs ``karnataka``), so we
  also keep a vowel-stripped "skeleton" of the name, which lines those two up.
"""

import csv
import os
import re
import sys
import unicodedata

from anyascii import anyascii

# csv fields in this dataset are short, but raise the limit so a malformed line can
# never abort a multi-hour run.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# --------------------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------------------

# Repository layout: this file lives at
#   student_resource/code/business_entity_resolution/src/common.py
# and the dataset at student_resource/dataset/. Resolve everything relative to that so
# the pipeline runs from any working directory.
_SRC = os.path.dirname(os.path.abspath(__file__))
STUDENT_ROOT = os.path.abspath(os.path.join(_SRC, "..", "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(STUDENT_ROOT, ".."))

# Every path is overridable by environment variable so the pipeline can run unchanged
# on a hosted notebook (SageMaker, Colab) where the dataset and the scratch volume are
# rarely siblings of the code. BER_WORK_DIR in particular matters: work/ peaks near
# 20GB, which is far larger than the root volume of a typical notebook instance.
CONFIG = {
    "data_dir": os.environ.get("BER_DATA_DIR", os.path.join(STUDENT_ROOT, "dataset")),
    "out_dir": os.environ.get("BER_OUT_DIR", os.path.join(STUDENT_ROOT, "output")),
    # Large intermediates (normalized shards, candidate arrays, features) live outside
    # the submission tree - they are reproducible and must not end up in the zip.
    "work_dir": os.environ.get("BER_WORK_DIR", os.path.join(PROJECT_ROOT, "work")),
    "seed": 42,
    # Candidate generation. TOP_K is per Source-1 entity over the *union* of S2 and S3
    # for its country. Ground truth averages 3.46 matches with a maximum of 11, so 40
    # leaves a wide recall margin while keeping the test candidate set near 50M pairs.
    "top_k": 40,
    "ngram_range": (4, 5),
    "ngram_min_df": 3,
    # Char n-grams are hashed instead of fitted into an explicit vocabulary: a fitted
    # vocabulary over 5M documents costs far more RAM than the 16GB budget allows.
    "n_features": 2 ** 20,
    "query_chunk": 4000,
    # A blocking token appearing in more than this many records is not distinctive
    # enough to be worth a posting list (measured: df<=2000 keeps recall high while
    # keeping postings short).
    "max_token_df": 2000,
    "max_posting": 400,
    # Training scale. The full 2.2M Source-1 training entities are unnecessary for a
    # ~30-feature GBDT and do not fit the time budget; these are sampled disjointly.
    "train_sample": 250_000,
    "holdout_sample": 50_000,
}

SOURCE_FILES = {
    ("train", 1): "train/train_source1.tsv",
    ("train", 2): "train/train_source2.tsv",
    ("train", 3): "train/train_source3.tsv",
    ("test", 1): "test/test_source1.tsv",
    ("test", 2): "test/test_source2.tsv",
    ("test", 3): "test/test_source3.tsv",
}
GROUND_TRUTH = "train/train_ground_truth.tsv"


def data_path(rel):
    return os.path.join(CONFIG["data_dir"], rel)


def work_path(*parts):
    p = os.path.join(CONFIG["work_dir"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def rss_gb():
    """Peak resident set size of this process, in GB.

    Every stage logs this. The pipeline is expected to run inside a 16GB machine, so a
    stage that quietly grows past a few GB is a bug to be caught while it is still
    small, not after the OS starts swapping.
    """
    import resource
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS/BSD and kilobytes on Linux.
    return m / (1024 ** 3) if sys.platform == "darwin" else m / (1024 ** 2)


def log(msg):
    """Timestamped progress line, flushed immediately.

    Stages run for tens of minutes, so buffered output would leave the operator with no
    idea whether the run is progressing or wedged.
    """
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Streaming TSV IO
# --------------------------------------------------------------------------------------

def stream_tsv(path):
    """Yield rows of a source TSV as lists of str, skipping the header.

    The source files are up to 500MB; ``pd.read_csv`` on all of them at once does not
    fit alongside the rest of the pipeline, so every full pass over raw data streams.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if row:
                yield row


def read_ground_truth(path=None):
    """Return {source1_entity_id: [matched ids]} from the ground-truth TSV."""
    path = path or data_path(GROUND_TRUTH)
    gt = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            matched = row[1] if len(row) > 1 else ""
            gt[row[0]] = [x for x in matched.split(",") if x]
    return gt


# --------------------------------------------------------------------------------------
# Token canonicalization tables
# --------------------------------------------------------------------------------------

# Legal-suffix variants are collapsed onto one spelling so that "Pvt Ltd", "Private
# Limited" and the romanized "praivet limited" all reduce to the same tokens.
LEGAL_SUFFIX = {
    "corporation": "corp", "corporated": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "limitd": "ltd",
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt",
    "praivet": "pvt",           # anyascii romanization of प्राइवेट
    "company": "co", "co": "co", "compant": "co",
    "llc": "llc", "llp": "llp", "elelpi": "llp",  # एलएलपी -> elelpi
    "plc": "plc", "sarl": "sarl", "sas": "sas", "sa": "sa", "eurl": "eurl",
    "gmbh": "gmbh", "and": "&", "et": "&",
}

# Street-type variants. "saint" appears because some records expand the abbreviation
# "ST" to the wrong word ("FREMONT SAINT" for "Fremont Street") - a real pattern in
# this data, not a hypothetical one.
STREET_ABBR = {
    "road": "rd", "rd": "rd",
    "street": "st", "st": "st", "saint": "st", "str": "st",
    "avenue": "ave", "ave": "ave", "av": "ave",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd", "boulevar": "blvd",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "terrace": "ter", "ter": "ter",
    "highway": "hwy", "hwy": "hwy",
    "square": "sq", "sq": "sq",
    "parkway": "pkwy", "pkwy": "pkwy",
    "circle": "cir", "cir": "cir",
    "trail": "trl", "trl": "trl",
    "apartment": "apt", "apt": "apt",
    "suite": "ste", "ste": "ste",
    "building": "bldg", "bldg": "bldg",
    "block": "blk", "blk": "blk",
    "cross": "crs", "main": "main", "sector": "sec", "sec": "sec",
    "nagar": "ngr", "ngr": "ngr",
    "colony": "cly", "cly": "cly",
    "rue": "rue", "boulevard_fr": "blvd",
}

# US states and Indian states, both directions: real matched pairs contain "NY" vs
# "New York" and "TN" vs "Tamil Nadu" vs the romanized "tmilnadu".
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
IN_STATES = {
    "andhra": "ap", "telangana": "tg", "karnataka": "ka", "krnatk": "ka",
    "kerala": "kl", "tamilnadu": "tn", "tmilnadu": "tn",
    "maharashtra": "mh", "mharastr": "mh",
    "gujarat": "gj", "gujrat": "gj", "rajasthan": "rj", "rajsthan": "rj",
    "punjab": "pb", "haryana": "hr", "hriyana": "hr",
    "delhi": "dl", "dilli": "dl",
    "uttarpradesh": "up", "uttrprdes": "up", "uttrpradesh": "up",
    "madhyapradesh": "mp", "mdhyprdes": "mp",
    "westbengal": "wb", "pscimbngal": "wb",
    "bihar": "br", "odisha": "or", "orissa": "or", "jharkhand": "jh",
    "assam": "as", "chhattisgarh": "cg", "uttarakhand": "uk", "goa": "ga",
    "himachalpradesh": "hp", "jammu": "jk", "kashmir": "jk",
}
STATE_MAP = dict(US_STATES)
STATE_MAP.update(IN_STATES)

# Tokens that carry no identity information. "null" is a literal string in this data,
# not a missing value; it appears inside addresses as a placeholder.
STOP_TOKENS = {
    "null", "none", "na", "nil", "po", "box", "pobox", "unit", "no", "nos",
    "door", "doorno", "near", "opp", "opposite", "behind", "the", "of", "at",
    "shop", "floor", "flr", "ph", "phone", "mob", "mobile", "tel",
}

_VOWELS = re.compile(r"[aeiou]")
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th|s|e|er|eme)$")
_ALNUM_SPLIT = re.compile(r"(\d+)")
_LONGNUM = re.compile(r"^\d{7,}$")  # phone numbers embedded in names/addresses
_ASCII_OK = re.compile(r"^[\x00-\x7F]*$")


def _fold(s):
    """Accent-strip, romanize non-Latin scripts, lowercase, and strip punctuation."""
    if not s:
        return ""
    if not _ASCII_OK.match(s):
        # NFKD + combining-mark removal handles Latin accents (Énterprises -> Enterprises)
        # cheaply; anyascii handles Devanagari/Tamil/Kannada/Bengali. anyascii is pure
        # Python, so it only runs on the ~7% of rows that need it.
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        if not _ASCII_OK.match(s):
            s = anyascii(s)
    s = s.lower()
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _canon_tokens(text, mapping, drop_stop=True):
    """Fold text to tokens, then canonicalize each token through ``mapping``.

    Alphanumeric runs like "af0684" are split into "af" + "684" and leading zeros are
    dropped, because the same address appears as both "Af-684" and "AF-0684".
    """
    out = []
    for tok in _fold(text).split():
        if drop_stop and tok in STOP_TOKENS:
            continue
        if _LONGNUM.match(tok):
            continue  # phone number, not an address component
        m = _ORDINAL.match(tok)
        if m:
            tok = m.group(1)
        # split mixed alnum ("af0684" -> "af","684"); keeps house numbers comparable
        parts = [p for p in _ALNUM_SPLIT.split(tok) if p]
        for p in parts:
            if p.isdigit():
                p = p.lstrip("0") or "0"
            else:
                p = mapping.get(p, p)
                p = STATE_MAP.get(p, p)
            if p:
                out.append(p)
    return out


def skeleton(text):
    """Vowel-stripped consonant skeleton of a name.

    Indic romanization systematically loses vowels, so "karnataka" and its romanized
    form "krnatk" look unrelated to an edit-distance metric but share the skeleton
    "krntk". This is the single cheapest signal for the transliterated ~7% of records.
    """
    s = _VOWELS.sub("", text.replace(" ", ""))
    return s


def normalize_record(name, addr):
    """Normalize one record into the fields every later stage consumes.

    Returns (name_norm, addr_norm, skeleton, digits) where ``digits`` is a
    space-joined string of the numeric components (house numbers, PIN/ZIP codes).
    """
    name_toks = _canon_tokens(name, LEGAL_SUFFIX, drop_stop=False)
    addr_toks = _canon_tokens(addr, STREET_ABBR, drop_stop=True)
    name_norm = " ".join(name_toks)
    addr_norm = " ".join(addr_toks)
    digits = " ".join(sorted({t for t in addr_toks if t.isdigit()}))
    return name_norm, addr_norm, skeleton(name_norm), digits


def f_beta(precision, recall, beta=0.5):
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f05(pred_map, gt_map):
    """Macro-averaged per-entity F_0.5, exactly as the challenge scores it.

    Both arguments map a Source-1 id to a *set* of matched ids. Every id in ``gt_map``
    contributes one term: an entity with no true matches scores 1.0 when the prediction
    is empty and 0.0 otherwise, which is why singletons are worth chasing.
    """
    total = 0.0
    for s1, truth in gt_map.items():
        pred = pred_map.get(s1, set())
        if not truth and not pred:
            total += 1.0
            continue
        if not truth or not pred:
            continue  # scores 0.0
        tp = len(pred & truth)
        if tp == 0:
            continue
        total += f_beta(tp / len(pred), tp / len(truth))
    return total / max(len(gt_map), 1)
