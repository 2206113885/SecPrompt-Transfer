"""Outcome-free construct audit for the frozen SecReq -> GitReq study.

This script reads public dataset text/labels only. It never calls a model or
consults target predictions. Its JSON output is a pre-deployment gate, not a
claim that the two corpora instantiate identical security concepts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


EXPECTED = {
    "secreq": "d6d1535bacb754f3390a60638f9de38bacd72499f694f34b14808b01b5c77083",
    "nfr": "8c9927bd648d41d6288ca40bcfd96e8e1e8cfb20d2b6eae1824c1057405bb90f",
    "fr": "9f5ee7490981803e8908890c7459bb63a20b591c956395b29741ed082bb560da",
}
EXPECTED_COUNTS = {
    "Security": 1646,
    "Performance": 1509,
    "Portability": 1284,
    "Availability": 738,
    "Functional": 531,
    "Fault-tolerance": 293,
    "Scalability": 157,
    "Maintainability": 144,
}
# Literal class-name shortcut. Freeze before model outcomes; do not expand post hoc.
EXPLICIT_LABEL_WORD = re.compile(r"\bsecurity\b", re.IGNORECASE)
# Broader predeclared sensitivity family, not a synonym for all security terms.
SECURE_FAMILY = re.compile(
    r"\b(?:security|secure|secures|secured|securing|insecure|insecurity|cybersecurity)\b",
    re.IGNORECASE,
)
FORMAL_MARKER = re.compile(
    r"\b(?:shall|must|will|user can|users can)\b|\bas a [^,.]{1,50},? i want\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def repository_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.lower() != "github.com" or len(parts) < 4 or parts[2] != "issues":
        raise ValueError(f"Unexpected issue URL: {url}")
    return f"{parts[0].lower()}/{parts[1].lower()}"


def near_duplicate_audit(
    source_texts: list[str], target_texts: list[str]
) -> dict[str, object]:
    """Global char-5-gram TF-IDF nearest-neighbor audit, without model outcomes."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(5, 5), lowercase=False, dtype=np.float32
    )
    matrix = vectorizer.fit_transform(source_texts + target_texts)
    source_matrix = matrix[: len(source_texts)]
    target_matrix = matrix[len(source_texts) :]
    maximum = np.zeros(len(target_texts), dtype=np.float32)
    closest_source = np.full(len(target_texts), -1, dtype=np.int32)
    for start in range(0, len(source_texts), 16):
        similarities = (source_matrix[start : start + 16] @ target_matrix.T).tocsr()
        for local_row in range(similarities.shape[0]):
            row = similarities.getrow(local_row)
            for target_index, value in zip(row.indices, row.data):
                if value > maximum[target_index]:
                    maximum[target_index] = value
                    closest_source[target_index] = start + local_row
    top_target = np.argsort(maximum)[-10:][::-1]
    return {
        "method": "global char-5-gram TF-IDF cosine; NFKC+casefold+whitespace normalization",
        "threshold_target_row_counts": {
            "0.85": int(np.count_nonzero(maximum >= 0.85)),
            "0.90": int(np.count_nonzero(maximum >= 0.90)),
            "0.95": int(np.count_nonzero(maximum >= 0.95)),
        },
        "maximum_similarity": float(maximum.max()),
        "top_pairs": [
            {
                "target_row_index_zero_based": int(target_index),
                "source_row_index_zero_based": int(closest_source[target_index]),
                "cosine": float(maximum[target_index]),
            }
            for target_index in top_target
        ],
    }


def audit(secreq: Path, nfr: Path, fr: Path, with_near_duplicates: bool = False) -> dict[str, object]:
    observed_hashes = {
        "secreq": sha256(secreq), "nfr": sha256(nfr), "fr": sha256(fr)
    }
    source = read_csv(secreq)
    nfr_rows = read_csv(nfr)
    fr_rows = read_csv(fr)
    target = nfr_rows + fr_rows
    labels = Counter(str(row["class_label"]) for row in target)
    source_labels = Counter(str(row["label"]) for row in source)
    source_texts = {normalize(row["text"]) for row in source}
    target_texts = [normalize(row["RequirementText"]) for row in target]
    urls = [row["url"] for row in target]
    repos = [repository_from_url(url) for url in urls]

    by_label: dict[str, dict[str, int]] = {}
    for label in EXPECTED_COUNTS:
        rows = [row for row in target if row["class_label"] == label]
        by_label[label] = {
            "n": len(rows),
            "literal_security": sum(bool(EXPLICIT_LABEL_WORD.search(row["RequirementText"])) for row in rows),
            "secure_family": sum(bool(SECURE_FAMILY.search(row["RequirementText"])) for row in rows),
            "formal_marker": sum(bool(FORMAL_MARKER.search(row["RequirementText"])) for row in rows),
        }
    source_by_label = {}
    for label in ("Security", "Non-security"):
        rows = [row for row in source if row["label"] == label]
        source_by_label[label] = {
            "n": len(rows),
            "literal_security": sum(bool(EXPLICIT_LABEL_WORD.search(row["text"])) for row in rows),
            "secure_family": sum(bool(SECURE_FAMILY.search(row["text"])) for row in rows),
            "formal_marker": sum(bool(FORMAL_MARKER.search(row["text"])) for row in rows),
        }
    nfr_labels = Counter(row["class_label"] for row in nfr_rows)
    fr_labels = Counter(row["class_label"] for row in fr_rows)
    checks = {
        "source_file_hash": observed_hashes["secreq"] == EXPECTED["secreq"],
        "nfr_file_hash": observed_hashes["nfr"] == EXPECTED["nfr"],
        "fr_file_hash": observed_hashes["fr"] == EXPECTED["fr"],
        "source_label_counts": source_labels == {"Security": 187, "Non-security": 323},
        "gitreq_label_counts": labels == EXPECTED_COUNTS,
        "nfr_fr_partition": len(nfr_rows) == 5771 and len(fr_rows) == 531
        and set(nfr_labels) == set(EXPECTED_COUNTS) - {"Functional"}
        and fr_labels == {"Functional": 531},
        "no_missing_text_url_or_label": all(
            row.get("RequirementText", "").strip()
            and row.get("class_label", "").strip()
            and row.get("url", "").strip()
            for row in target
        ),
        "no_exact_source_target_overlap": not any(text in source_texts for text in target_texts),
        "repository_count": len(set(repos)) == 4080,
        "issue_url_count": len(set(urls)) == 6140,
    }
    result = {
        "status": "PASS_WITH_CONSTRUCT_LIMITS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "file_sha256": observed_hashes,
        "normalization": "NFKC + casefold + whitespace collapse",
        "label_word_regex": EXPLICIT_LABEL_WORD.pattern,
        "secure_family_regex": SECURE_FAMILY.pattern,
        "formal_marker_regex": FORMAL_MARKER.pattern,
        "source_rows": len(source),
        "source_by_label": source_by_label,
        "gitreq_rows": len(target),
        "nfr_rows": len(nfr_rows),
        "fr_rows": len(fr_rows),
        "target_by_label": by_label,
        "t_nfr": {"rows": len(nfr_rows), "security": nfr_labels["Security"], "nonsecurity": len(nfr_rows) - nfr_labels["Security"]},
        "t_all": {"rows": len(target), "security": labels["Security"], "nonsecurity": len(target) - labels["Security"]},
        "unique_repositories": len(set(repos)),
        "unique_issue_urls": len(set(urls)),
        "normalized_exact_overlap_rows": sum(text in source_texts for text in target_texts),
        "normalized_target_duplicate_extra_rows": len(target_texts) - len(set(target_texts)),
        "url_multirow_count": sum(count > 1 for count in Counter(urls).values()),
        "repository_multirow_count": sum(count > 1 for count in Counter(repos).values()),
        "known_limitations": [
            "GitReq includes only category/intent/keyword-mined GitHub issues, not a prevalence sample of all issues.",
            "T_NFR narrows the SecReq negative construct by removing functional requirements.",
            "T_All includes functional rows selected with a distinct formal-language pipeline.",
            "Gold eight-way labels were expert validated, but the binary mapping was not independently re-annotated.",
            "Previous GitReq model pilots exist; the formal deployment can only be prospectively frozen from this point.",
        ],
    }
    if with_near_duplicates:
        result["cross_corpus_near_duplicates"] = near_duplicate_audit(
            [normalize(row["text"]) for row in source], target_texts
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secreq", type=Path, required=True)
    parser.add_argument("--nfr", type=Path, required=True)
    parser.add_argument("--fr", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--near-duplicates", action="store_true")
    args = parser.parse_args()
    result = audit(args.secreq, args.nfr, args.fr, args.near_duplicates)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print("GITREQ_CONSTRUCT_AUDIT " + json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
