# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub==1.*",
#     "pandas==2.*",
#     "scikit-learn==1.*",
#     "datasketch==1.*",
# ]
# ///
"""Profile the Hugging Face dataset Tobi-Bueck/customer-support-tickets.

Run:  uv run profile.py            (downloads into data/raw, writes the outputs)
      uv run profile.py --revision main   (profile the newest commit instead)

Outputs (in the repository root):
  profile_report.md        the report
  reading_sample_100.csv   100 English tickets to read (labels hidden)
  reading_sample_key.csv   the hidden labels for those 100 tickets

No AI model is called. Every random step uses seed 42.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from huggingface_hub import HfApi, snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

REPO_ID = "Tobi-Bueck/customer-support-tickets"
# Commit profiled for the committed report. Pass --revision main to use the newest one.
DEFAULT_REVISION = "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"
SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
REPORT = ROOT / "profile_report.md"
SAMPLE_CSV = ROOT / "reading_sample_100.csv"
KEY_CSV = ROOT / "reading_sample_key.csv"

LABELS = ["queue", "priority", "type"]
TEXT_FIELDS = ["subject", "body", "answer"]
SHINGLE = 5
NUM_PERM = 128
NEAR_THRESHOLD = 0.8
SAMPLE_PER_QUEUE = 20
READING_N = 100
READING_LANGS = ["en"]  # user choice: English only

# What the dataset card (README.md) says, typed in by hand from the card so the
# script can compare it with the data. Section 10 lists every mismatch.
CARD = {
    "n_versions": 2,
    "fields": ["queue", "priority", "language", "subject", "body", "answer", "type",
               "business_type", "tags"],
    "n_tag_columns": 10,
    "queue": ["Technical Support", "Customer Service", "Billing and Payments",
              "Product Support", "IT Support", "Returns and Exchanges",
              "Sales and Pre-Sales", "Human Resources",
              "Service Outages and Maintenance", "General Inquiry"],
    "priority": ["low", "medium", "critical"],  # card: 1 Low, 2 Medium, 3 Critical
    "type": ["Incident", "Request", "Problem", "Change"],
    "language": ["en", "de"],
}

READABLE_EXT = {".csv": ",", ".tsv": "\t", ".json": None, ".jsonl": None}
META_FILES = {"README.md", ".gitattributes"}

PLACEHOLDER_PATTERNS = {
    "{curly}": re.compile(r"\{[^{}\n]{1,40}\}"),
    "<angle>": re.compile(r"<[^<>\n]{1,40}>"),
    "[square]": re.compile(r"\[[^\[\]\n]{1,40}\]"),
}

DE_MARKERS = {"und", "nicht", "ich", "wir", "ist", "der", "die", "das", "mit", "für",
              "bitte", "sehr", "ihre", "eine", "haben", "können"}
EN_MARKERS = {"the", "and", "is", "please", "with", "for", "you", "we", "our", "have",
              "can", "this", "that", "your", "are", "would"}


# --------------------------------------------------------------------------- helpers

def md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a Markdown table without extra libraries."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        s = str(v).replace("|", "\\|").replace("\n", " ")
        return s

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def norm_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def fence(text: str) -> str:
    return "~~~~text\n" + ("" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)) + "\n~~~~"


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------- load

def download(revision: str) -> tuple[str, list[dict]]:
    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    snapshot_download(REPO_ID, repo_type="dataset", revision=info.sha, local_dir=str(DATA_DIR))
    files = [{"file": s.rfilename, "size_bytes": s.size} for s in info.siblings]
    return info.sha, files


def load_files(files: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    frames, inventory = [], []
    for f in files:
        name = f["file"]
        ext = Path(name).suffix.lower()
        entry = dict(f)
        if name in META_FILES:
            entry["status"] = "metadata (not data)"
        elif ext not in READABLE_EXT:
            entry["status"] = f"NOT LOADED: no reader for '{ext}' with the allowed libraries"
        else:
            path = DATA_DIR / name
            if ext in (".csv", ".tsv"):
                d = pd.read_csv(path, sep=READABLE_EXT[ext], dtype=str, keep_default_na=True)
            else:
                d = pd.read_json(path, lines=(ext == ".jsonl"), dtype=False)
                d = d.astype("string").astype(object)
            # Keep the native dtypes for section 2.
            if ext in (".csv", ".tsv"):
                native = pd.read_csv(path, sep=READABLE_EXT[ext])
            else:
                native = pd.read_json(path, lines=(ext == ".jsonl"))
            entry["status"] = "loaded"
            entry["rows"] = len(d)
            entry["columns"] = list(d.columns)
            entry["native_dtypes"] = {c: str(t) for c, t in native.dtypes.items()}
            d.insert(0, "row", np.arange(len(d)))
            d.insert(0, "file", name)
            frames.append(d)
        inventory.append(entry)
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["subject", "body", "answer", "queue", "priority", "type", "language", "version"]:
        if c not in df.columns:
            df[c] = np.nan
    df["id"] = [
        "t" + hashlib.sha1(f"{f}#{r}".encode()).hexdigest()[:10] for f, r in zip(df["file"], df["row"])
    ]
    df["version_label"] = df["version"].where(df["version"].notna(), "none (no version column)")
    df["text"] = df["subject"].fillna("") + "\n" + df["body"].fillna("")
    return df, inventory


# --------------------------------------------------------------------------- duplicates

def cluster_duplicates(df: pd.DataFrame) -> dict:
    """Exact and near-duplicate detection on subject + body, over all files pooled."""
    normed = norm_text(df["subject"]) + " || " + norm_text(df["body"])
    df["norm_text"] = normed
    df["exact_group"] = normed.factorize()[0]

    # MinHash on 5-character shingles. One signature per distinct normalised text.
    uniq = normed.drop_duplicates()
    uniq_texts = uniq.tolist()
    shingle_sets = []
    for t in uniq_texts:
        if len(t) < SHINGLE:
            shingle_sets.append([t.encode("utf8")])
        else:
            shingle_sets.append(list({t[i:i + SHINGLE].encode("utf8") for i in range(len(t) - SHINGLE + 1)}))
    mhs = MinHash.bulk(shingle_sets, num_perm=NUM_PERM, seed=SEED)
    lsh = MinHashLSH(threshold=NEAR_THRESHOLD, num_perm=NUM_PERM)
    for i, m in enumerate(mhs):
        lsh.insert(i, m)
    uf = UnionFind(len(mhs))
    n_pairs = 0
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            if j > i and m.jaccard(mhs[j]) >= NEAR_THRESHOLD:  # confirm LSH candidates
                uf.union(i, j)
                n_pairs += 1
    text_to_uniq = {t: i for i, t in enumerate(uniq_texts)}
    roots = [uf.find(text_to_uniq[t]) for t in normed]
    # Cluster id numbered by first appearance.
    order = {}
    for r in roots:
        order.setdefault(r, len(order))
    df["cluster_id"] = [f"c{order[r]:06d}" for r in roots]
    return {"near_pairs": n_pairs, "n_unique_norm": len(uniq_texts)}


def dup_rates(sub: pd.DataFrame) -> dict:
    n = len(sub)
    ex_sizes = sub.groupby("norm_text").size()
    cl_sizes = sub.groupby("cluster_id").size()
    return {
        "rows": n,
        "exact_distinct": len(ex_sizes),
        "exact_dup_rate": 1 - len(ex_sizes) / n,  # share of rows that repeat an earlier row
        "rows_in_exact_group>1": float((ex_sizes[ex_sizes > 1]).sum() / n),
        "near_clusters": len(cl_sizes),
        "near_dup_rate": 1 - len(cl_sizes) / n,
        "rows_in_cluster>1": float((cl_sizes[cl_sizes > 1]).sum() / n),
    }


# --------------------------------------------------------------------------- baselines

def run_baselines(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fname, fdf in df.groupby("file", sort=True):
        for lang in ["en", "de"]:
            sub = fdf[fdf["language"] == lang]
            if len(sub) < 50:
                continue
            for label in LABELS:
                y = sub[label]
                if y.isna().all():
                    rows.append({"file": fname, "language": lang, "label": label,
                                 "note": "label column missing in this file"})
                    continue
                s = sub[y.notna()]
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                tr, te = next(gss.split(s, groups=s["cluster_id"]))
                s_tr, s_te = s.iloc[tr], s.iloc[te]
                assert not set(s_tr["cluster_id"]) & set(s_te["cluster_id"])
                maj = s_tr[label].value_counts().idxmax()
                y_maj = np.full(len(s_te), maj, dtype=object)
                vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                                      max_features=200_000)
                X_tr = vec.fit_transform(s_tr["text"])
                X_te = vec.transform(s_te["text"])
                clf = LogisticRegression(max_iter=3000, random_state=SEED)
                clf.fit(X_tr, s_tr[label])
                y_hat = clf.predict(X_te)
                rows.append({
                    "file": fname, "language": lang, "label": label,
                    "classes": s[label].nunique(), "train": len(s_tr), "test": len(s_te),
                    "test_share": len(s_te) / len(s),
                    "majority_class": maj,
                    "maj_acc": accuracy_score(s_te[label], y_maj),
                    "maj_macroF1": f1_score(s_te[label], y_maj, average="macro", zero_division=0),
                    "lr_acc": accuracy_score(s_te[label], y_hat),
                    "lr_macroF1": f1_score(s_te[label], y_hat, average="macro", zero_division=0),
                    "note": "",
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reading sample

def reading_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    pool = df[df["language"].isin(READING_LANGS)].copy()
    # One row per duplicate cluster so the reader never sees the same ticket twice.
    pool = pool.sample(frac=1, random_state=SEED).drop_duplicates("cluster_id")
    pool["len_chars"] = pool["text"].str.len()
    pool["length_third"] = pd.qcut(pool["len_chars"].rank(method="first"), 3,
                                   labels=["short", "medium", "long"]).astype(str)
    pool["stratum"] = pool["version_label"].astype(str) + " | " + pool["length_third"]
    groups = {k: g for k, g in pool.groupby("stratum", sort=True)}
    keys = sorted(groups)
    # Equal share per group; if a group is too small, give its remainder to the others.
    alloc = {k: 0 for k in keys}
    left = READING_N
    while left > 0:
        open_keys = [k for k in keys if alloc[k] < len(groups[k])]
        if not open_keys:
            break
        share = max(left // len(open_keys), 1)
        for k in rng.permutation(open_keys):
            if left == 0:
                break
            take = min(share, len(groups[k]) - alloc[k], left)
            alloc[k] += take
            left -= take
    picked = pd.concat([groups[k].sample(n=alloc[k], random_state=SEED) for k in keys if alloc[k]])
    picked = picked.sample(frac=1, random_state=SEED)  # order gives no hint of stratum
    sample = picked[["id", "subject", "body"]].copy()
    sample["note"] = ""
    key = picked[["id", "file", "row", "version_label", "language", "length_third", "len_chars",
                  "queue", "priority", "type", "answer", "cluster_id"]].copy()
    alloc_df = pd.DataFrame({"stratum": keys, "available": [len(groups[k]) for k in keys],
                             "sampled": [alloc[k] for k in keys]})
    return sample, key, alloc_df


# --------------------------------------------------------------------------- report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    args = ap.parse_args()

    sha, files = download(args.revision)
    df, inventory = load_files(files)
    loaded = [e for e in inventory if e["status"] == "loaded"]
    out: list[str] = []
    w = out.append

    w(f"# Profile: `{REPO_ID}`\n")
    w(f"- Commit profiled: `{sha}`")
    w(f"- Rows loaded (all data files): {len(df):,}")
    w(f"- Seed: {SEED}. No AI model was called. Generated by `profile.py`.\n")

    # ---- 1
    w("## 1. Files and versions\n")
    inv = pd.DataFrame([{"file": e["file"], "size_bytes": e["size_bytes"], "status": e["status"],
                         "rows": e.get("rows", ""), "columns": len(e.get("columns", [])) or ""}
                        for e in inventory])
    w(md_table(inv) + "\n")
    w("Rows per file, per version, per language (`version` exists only where the file has that column):\n")
    t = df.groupby(["file", "version_label", "language"], dropna=False).size().rename("rows")
    w(md_table(t.reset_index()) + "\n")
    t2 = pd.crosstab([df["file"], df["version_label"]], df["language"].fillna("NA"), margins=True)
    w(md_table(t2, index=True) + "\n")

    # ---- 2
    w("## 2. Columns, types and null rates\n")
    w("`native dtype` is what pandas infers. `null` is a missing value; `empty` is an empty or whitespace-only string.\n")
    for e in loaded:
        d = df[df["file"] == e["file"]][e["columns"]]
        rows = []
        for c in e["columns"]:
            s = d[c]
            empt = s.notna() & s.astype(str).str.strip().eq("")
            rows.append({"column": c, "native dtype": e["native_dtypes"][c], "non-null": int(s.notna().sum()),
                         "null": int(s.isna().sum()), "null_rate": pct(s.isna().mean()),
                         "empty": int(empt.sum()), "distinct": int(s.nunique())})
        w(f"### `{e['file']}` ({e['rows']:,} rows)\n")
        w(md_table(pd.DataFrame(rows)) + "\n")

    # ---- 3
    w("## 3. Distinct values: queue, priority, type, version\n")
    for col in ["queue", "priority", "type", "version"]:
        w(f"### {col}\n")
        vc = pd.crosstab(df[col].fillna("<missing>"), df["file"], margins=True, margins_name="ALL")
        vc = vc.sort_values("ALL", ascending=False)
        w(md_table(vc, index=True) + "\n")

    # ---- 4
    w("## 4. Cross-tab queue x priority x type (top 50 cells)\n")
    for fname, fdf in df.groupby("file", sort=True):
        w(f"### `{fname}`\n")
        if fdf["type"].isna().all():
            w("This file has no `type` column, so the cross-tab is queue x priority (top 50).\n")
            ct = fdf.groupby(["queue", "priority"]).size()
        else:
            ct = fdf.groupby(["queue", "priority", "type"]).size()
        ct = ct.sort_values(ascending=False).rename("rows").reset_index()
        ct["share"] = (ct["rows"] / len(fdf)).map(pct)
        w(f"{len(ct)} non-empty cells in total.\n")
        w(md_table(ct.head(50)) + "\n")

    # ---- 5
    w("## 5. Text length distribution (characters and words)\n")
    qs = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    for unit in ["chars", "words"]:
        w(f"### {unit}\n")
        rows = []
        for (fname, lang), g in df.groupby(["file", "language"], sort=True):
            for fld in TEXT_FIELDS:
                s = g[fld].dropna().astype(str)
                if s.empty:
                    continue
                L = s.str.len() if unit == "chars" else s.str.split().str.len()
                q = L.quantile(qs).round(0).astype(int).tolist()
                rows.append({"file": fname, "lang": lang, "field": fld, "n": len(L),
                             "min": q[0], "p5": q[1], "p25": q[2], "median": q[3], "p75": q[4],
                             "p95": q[5], "p99": q[6], "max": q[7], "mean": round(L.mean(), 1)})
        w(md_table(pd.DataFrame(rows), floatfmt="{:.1f}") + "\n")
    w("Histogram of character length, all files pooled (missing values excluded):\n")
    bins = [0, 25, 50, 100, 200, 400, 800, 1600, 3200, np.inf]
    hist = {fld: pd.cut(df[fld].dropna().astype(str).str.len(), bins, right=False).value_counts().sort_index()
            for fld in TEXT_FIELDS}
    hdf = pd.DataFrame(hist)
    hdf.index = hdf.index.astype(str)
    w(md_table(hdf, index=True) + "\n")

    # ---- 6
    w("## 6. Duplicates\n")
    dinfo = cluster_duplicates(df)
    w(f"Text compared: `subject + body`, lower-cased, runs of whitespace replaced by one space. "
      f"Near-duplicates: MinHash ({NUM_PERM} permutations, seed {SEED}) on {SHINGLE}-character "
      f"shingles, LSH threshold {NEAR_THRESHOLD}; each LSH candidate pair is kept only if its estimated "
      f"Jaccard is >= {NEAR_THRESHOLD}. Clusters are the connected groups of those pairs, so exact duplicates "
      f"are always in the same cluster. All files are clustered together, so a `cluster_id` "
      f"can span files. Confirmed near pairs: {dinfo['near_pairs']:,}.\n")
    w("`*_dup_rate` = 1 - (distinct groups / rows): the share of rows that are a repeat. "
      "`rows_in_*>1` = the share of rows that have at least one twin.\n")
    rows = [{"scope": "ALL files pooled", **dup_rates(df)}]
    for fname, g in df.groupby("file", sort=True):
        rows.append({"scope": fname, **dup_rates(g)})
        for lang, gl in g.groupby("language", sort=True):
            rows.append({"scope": f"{fname} [{lang}]", **dup_rates(gl)})
    drt = pd.DataFrame(rows)
    for c in ["exact_dup_rate", "rows_in_exact_group>1", "near_dup_rate", "rows_in_cluster>1"]:
        drt[c] = drt[c].map(pct)
    w(md_table(drt) + "\n")

    # Cross-file overlap
    files_per_cluster = df.groupby("cluster_id")["file"].nunique()
    multi = files_per_cluster[files_per_cluster > 1].index
    w("Overlap between files (rows whose cluster also contains a row from another file):\n")
    ov = df[df["cluster_id"].isin(multi)].groupby("file").size().rename("rows_shared").reset_index()
    ov["share_of_file"] = [pct(r / (df["file"] == f).sum()) for f, r in zip(ov["file"], ov["rows_shared"])]
    w(md_table(ov) + "\n" if len(ov) else "None.\n")
    pair_rows = []
    fl = sorted(df["file"].unique())
    cl_sets = {f: set(df.loc[df["file"] == f, "cluster_id"]) for f in fl}
    for i, a in enumerate(fl):
        for b in fl[i + 1:]:
            pair_rows.append({"file_a": a, "file_b": b, "shared_clusters": len(cl_sets[a] & cl_sets[b])})
    w(md_table(pd.DataFrame(pair_rows)) + "\n")

    # Label conflicts inside clusters
    w("Duplicate clusters whose rows disagree on a label (clusters with >1 row and that label present):\n")
    conf = []
    big = df.groupby("cluster_id").filter(lambda g: len(g) > 1)
    for lab in LABELS:
        g = big[big[lab].notna()].groupby("cluster_id")[lab]
        sizes = g.size()
        nun = g.nunique()[sizes > 1]
        conf.append({"label": lab, "clusters_with_2+_labelled_rows": len(nun),
                     "clusters_with_conflict": int((nun > 1).sum()),
                     "conflict_rate": pct((nun > 1).mean()) if len(nun) else ""})
    conf_df = pd.DataFrame(conf)
    w(md_table(conf_df) + "\n")
    sizes = df.groupby("cluster_id").size()
    w("Largest clusters:\n")
    top = sizes.sort_values(ascending=False).head(10)
    tops = []
    for cid, n in top.items():
        r = df[df["cluster_id"] == cid].iloc[0]
        tops.append({"cluster_id": cid, "rows": n, "files": df.loc[df["cluster_id"] == cid, "file"].nunique(),
                     "example subject": str(r["subject"])[:60] if pd.notna(r["subject"]) else "(missing)",
                     "example body": str(r["body"])[:80]})
    w(md_table(pd.DataFrame(tops)) + "\n")
    w("Every row's `cluster_id` is in `reading_sample_key.csv` for sampled rows; the full row-to-cluster "
      "map is written to `data/cluster_ids.csv` (git-ignored with the data).\n")
    df[["id", "file", "row", "cluster_id"]].to_csv(DATA_DIR.parent / "cluster_ids.csv", index=False)

    # ---- 7
    w("## 7. Templates and placeholders\n")
    w("Patterns: `{...}`, `<...>`, `[...]` with 1-40 characters inside and no line break. "
      "Note: `<...>` also matches HTML-like tags; the token table shows what was actually found.\n")
    hit_any = pd.Series(False, index=df.index)
    per = []
    tokens = Counter()
    examples_pool = []
    for fld in TEXT_FIELDS:
        s = df[fld].fillna("").astype(str)
        fld_any = pd.Series(False, index=df.index)
        for pname, rx in PLACEHOLDER_PATTERNS.items():
            m = s.str.contains(rx)
            fld_any |= m
            per.append({"field": fld, "pattern": pname, "rows": int(m.sum()), "share": pct(m.mean())})
            for idx in df.index[m]:
                for tok in rx.findall(s[idx]):
                    tokens[tok] += 1
        hit_any |= fld_any
        per.append({"field": fld, "pattern": "ANY", "rows": int(fld_any.sum()), "share": pct(fld_any.mean())})
    w(f"**Rows with at least one placeholder in subject, body or answer: {int(hit_any.sum()):,} "
      f"of {len(df):,} ({pct(hit_any.mean())}).**\n")
    w(md_table(pd.DataFrame(per)) + "\n")
    byfile = df.assign(hit=hit_any).groupby(["file", "language"])["hit"].agg(["sum", "mean"]).reset_index()
    byfile["mean"] = byfile["mean"].map(pct)
    byfile.columns = ["file", "language", "rows_with_placeholder", "share"]
    w(md_table(byfile) + "\n")
    w("Most common placeholder tokens (top 40):\n")
    w(md_table(pd.DataFrame(tokens.most_common(40), columns=["token", "occurrences"])) + "\n")
    w("10 random examples:\n")
    ex = df[hit_any].sample(n=min(10, int(hit_any.sum())), random_state=SEED)
    for _, r in ex.iterrows():
        found = sorted({t for fld in TEXT_FIELDS for rx in PLACEHOLDER_PATTERNS.values()
                        for t in rx.findall(str(r[fld]) if pd.notna(r[fld]) else "")})
        w(f"**{r['id']}** (`{r['file']}`, {r['language']}) - placeholders: {', '.join('`'+x+'`' for x in found)}\n")
        for fld in TEXT_FIELDS:
            if pd.notna(r[fld]):
                w(f"*{fld}:*\n{fence(r[fld])}\n")

    # ---- 8
    w("## 8. 20 random full tickets per queue value\n")
    w("Rows drawn at random (seed 42) from all files pooled. Each queue is folded; click to open.\n")
    for q, g in sorted(df[df["queue"].notna()].groupby("queue"), key=lambda kv: -len(kv[1])):
        smp = g.sample(n=min(SAMPLE_PER_QUEUE, len(g)), random_state=SEED)
        w(f"<details><summary><b>{q}</b> ({len(g):,} rows; showing {len(smp)})</summary>\n")
        for _, r in smp.iterrows():
            tags = [str(r[c]) for c in df.columns if c.startswith("tag_") and pd.notna(r[c])]
            meta = (f"`{r['id']}` | file `{r['file']}` | version {r['version_label']} | lang {r['language']} | "
                    f"priority {r['priority']} | type {r['type'] if pd.notna(r['type']) else '-'} | "
                    f"cluster {r['cluster_id']} | tags: {', '.join(tags) if tags else '-'}")
            w(f"#### {meta}\n")
            for fld in TEXT_FIELDS:
                if fld in r and pd.notna(r[fld]):
                    w(f"*{fld}:*\n{fence(r[fld])}\n")
                elif fld == "subject" and pd.isna(r[fld]):
                    w("*subject:* (missing)\n")
        w("</details>\n")

    # ---- 9
    w("## 9. Baselines (English and German apart)\n")
    w("Input text: `subject + body` (the answer is not used). Each file is modelled on its own because "
      "label sets differ between files. Split: `GroupShuffleSplit`, 20% of duplicate clusters to test, "
      "seed 42, grouped by the pooled `cluster_id`, so no cluster is on both sides (checked by an assert). "
      "(a) majority class of the training split; (b) TF-IDF (word 1-2 grams, sublinear tf, min_df 2) + "
      "logistic regression (lbfgs, max_iter 3000, default C=1).\n")
    bl = run_baselines(df)
    show = bl.copy()
    for c in ["classes", "train", "test"]:
        if c in show:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
    for c in ["test_share", "maj_acc", "maj_macroF1", "lr_acc", "lr_macroF1"]:
        if c in show:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    w(md_table(show.fillna("")) + "\n")

    # ---- 10
    w("## 10. Surprises (what the dataset card does not describe)\n")
    surprises = []
    data_files = [e["file"] for e in loaded]
    surprises.append(f"The card speaks of {CARD['n_versions']} versions / files; the repository has "
                     f"{len(data_files)} data files: {', '.join('`'+f+'`' for f in data_files)}. "
                     f"No file is named on the card.")
    for e in inventory:
        if e["status"].startswith("NOT LOADED"):
            surprises.append(f"File `{e['file']}` could not be read with the allowed libraries.")
    card_fields = set(CARD["fields"])
    for e in loaded:
        cols = e["columns"]
        tag_cols = [c for c in cols if c.startswith("tag_")]
        extra = [c for c in cols if c not in card_fields and not c.startswith("tag_")]
        missing = [f for f in CARD["fields"] if f not in cols and not (f == "tags" and tag_cols)]
        if extra:
            surprises.append(f"`{e['file']}`: columns not on the card: {extra}.")
        if missing:
            surprises.append(f"`{e['file']}`: card fields absent from the file: {missing}.")
        if tag_cols and len(tag_cols) != CARD["n_tag_columns"]:
            surprises.append(f"`{e['file']}`: the card says tags are split into {CARD['n_tag_columns']} "
                             f"columns; the file has {len(tag_cols)} ({tag_cols[0]}..{tag_cols[-1]}).")
        for c in cols:
            nr = df.loc[df["file"] == e["file"], c].isna().mean()
            if c in ("subject", "body", "answer") and nr > 0:
                surprises.append(f"`{e['file']}`: `{c}` is missing in {pct(nr)} of rows.")
    for col in ["queue", "priority", "type", "language"]:
        for fname, g in df.groupby("file", sort=True):
            vals = set(g[col].dropna().unique())
            if not vals:
                continue
            ref = set(CARD[col])
            ref_l = {v.lower() for v in ref}
            unknown = sorted(v for v in vals if str(v).lower() not in ref_l)
            absent = sorted(v for v in ref if v.lower() not in {str(x).lower() for x in vals})
            if unknown:
                more = f" (showing 10 of {len(unknown)})" if len(unknown) > 10 else ""
                surprises.append(f"`{fname}`: `{col}` values not on the card{more}: {unknown[:10]}.")
            if absent:
                surprises.append(f"`{fname}`: `{col}` values on the card but not in the file: {absent}.")
    if df["version"].notna().any():
        vv = df.groupby("file")["version"].apply(lambda s: sorted(s.dropna().unique().tolist()))
        for f, v in vv.items():
            if v:
                surprises.append(f"`{f}` has a `version` column with values {v}; the card does not explain "
                                 f"these numbers or how they relate to the file names.")
    if df["priority"].dropna().str.match(r"^\d$").sum() == 0:
        surprises.append("The card describes priority as 1/2/3; the data uses words only.")
    for _, r in ov.iterrows():
        surprises.append(f"`{r['file']}`: {r['rows_shared']:,} rows ({r['share_of_file']}) are in a duplicate "
                         f"cluster that also contains rows of another file; the card does not say the files overlap.")
    for _, r in drt.iterrows():
        if r["scope"] in data_files:
            surprises.append(f"`{r['scope']}`: exact-duplicate rate {r['exact_dup_rate']}, near-duplicate "
                             f"rate {r['near_dup_rate']} (within the file).")
    for _, r in conf_df.iterrows():
        if r["clusters_with_conflict"]:
            surprises.append(f"{r['clusters_with_conflict']:,} duplicate clusters carry conflicting "
                             f"`{r['label']}` labels ({r['conflict_rate']} of clusters with 2+ labelled rows).")
    surprises.append(f"{int(hit_any.sum()):,} rows ({pct(hit_any.mean())}) contain placeholder-like tokens "
                     f"(top: {', '.join('`'+t+'`' for t, _ in tokens.most_common(5))}); the card does not "
                     f"mention anonymisation or templates.")
    # Crude language check (word markers, no model).
    words = df["text"].str.lower().str.findall(r"[a-zäöüß]+").map(set)
    de_hits = words.map(lambda s: len(s & DE_MARKERS))
    en_hits = words.map(lambda s: len(s & EN_MARKERS))
    w_in_lbl = []
    for (f, lang), g in df.assign(_de=de_hits, _en=en_hits).groupby(["file", "language"]):
        other = (g["_de"] >= 3) & (g["_de"] > g["_en"]) if lang == "en" else (g["_en"] >= 3) & (g["_en"] > g["_de"])
        if other.sum():
            w_in_lbl.append(f"`{f}` [{lang}]: {int(other.sum()):,} ({pct(other.mean())})")
    if w_in_lbl:
        surprises.append("Language label check (word-marker heuristic, no model) - rows whose subject + body "
                         "look like the other language: " + "; ".join(w_in_lbl) + ".")
    for fld in TEXT_FIELDS:
        lit = df[fld].fillna("").astype(str).str.contains(r"\\n", regex=True)
        if lit.sum():
            byf = df[lit].groupby("file").size().to_dict()
            surprises.append(f"`{fld}` contains a literal backslash-n (`\\n` as two characters, not a line break) "
                             f"in {int(lit.sum()):,} rows ({pct(lit.mean())}); by file: {byf}.")
    subj_eq_body = (norm_text(df["subject"]) == norm_text(df["body"])).sum()
    if subj_eq_body:
        surprises.append(f"{subj_eq_body:,} rows have subject identical to body.")
    for s in surprises:
        w(f"- {s}")
    w("")

    # ---- reading sample
    sample, key, alloc = reading_sample(df)
    sample.to_csv(SAMPLE_CSV, index=False)
    key.to_csv(KEY_CSV, index=False)
    w("## Appendix: reading sample\n")
    w(f"`{SAMPLE_CSV.name}`: {len(sample)} tickets, language {READING_LANGS}, one row per duplicate cluster. "
      "Strata = version x text-length third (thirds of `subject + body` character length, computed on the "
      "English pool). Equal share per stratum, random inside each (seed 42). Rows from a file without a "
      "`version` column form their own stratum `none`. Labels and answers are in "
      f"`{KEY_CSV.name}`.\n")
    w(md_table(alloc) + "\n")

    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {REPORT.name} ({REPORT.stat().st_size:,} bytes), {SAMPLE_CSV.name}, {KEY_CSV.name}")


if __name__ == "__main__":
    main()
