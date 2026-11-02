"""Real dataset auditing module for Superviz26-SQL.

Produces comprehensive data audit covering row counts, partition distributions,
missing values, duplicate queries, template IDs, attack families, and confounding checks.
"""

import csv
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.loader import CANONICAL_COLUMNS, REQUIRED_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_sql(q: str) -> str:
    """Normalize SQL statement for duplicate collision detection."""
    s = q.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([;,()=])\s*", r"\1", s)
    return s.strip()


def audit_csv_file(file_path: Path, max_rows: int = None) -> dict[str, Any]:
    """Audit a single CSV file in the dataset."""
    logger.info(f"Auditing file {file_path} ({file_path.stat().st_size:,} bytes)...")
    start_time = time.time()

    col_counts = Counter()
    missing_counts = Counter()
    label_counts = Counter()
    split_counts = Counter()
    partition_label_matrix = Counter()  # (split, label)
    statement_types = Counter()
    attack_techniques = Counter()
    attack_stages = Counter()
    attack_statuses = Counter()
    tamper_methods = Counter()
    unique_templates = set()
    unique_attacks = set()

    raw_queries_seen = set()
    norm_queries_seen = set()
    query_to_labels: dict[str, set[int]] = {}

    exact_duplicates = 0
    norm_duplicates = 0
    total_rows = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"error": f"Empty file: {file_path}"}

        col_to_idx = {col.strip(): i for i, col in enumerate(header)}
        for c in header:
            col_counts[c] += 1

        for row_idx, row in enumerate(reader, start=2):
            if not row or all(c == "" for c in row):
                continue
            total_rows += 1

            # Pad row if needed
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))

            full_query = row[col_to_idx["full_query"]] if "full_query" in col_to_idx else ""
            raw_label = row[col_to_idx["label"]] if "label" in col_to_idx else ""
            raw_split = row[col_to_idx["split"]] if "split" in col_to_idx else ""
            stmt = row[col_to_idx["statement_type"]] if "statement_type" in col_to_idx else ""
            tech = row[col_to_idx["attack_technique"]] if "attack_technique" in col_to_idx else ""
            stage = row[col_to_idx["attack_stage"]] if "attack_stage" in col_to_idx else ""
            status = row[col_to_idx["attack_status"]] if "attack_status" in col_to_idx else ""
            tamper = row[col_to_idx["tamper_method"]] if "tamper_method" in col_to_idx else ""
            tmpl = row[col_to_idx["query_template_id"]] if "query_template_id" in col_to_idx else ""
            atk_id = row[col_to_idx["attack_id"]] if "attack_id" in col_to_idx else ""

            # Check missing
            for col, idx in col_to_idx.items():
                if idx >= len(row) or row[idx] == "":
                    missing_counts[col] += 1

            # Label parsing
            try:
                lbl = int(raw_label.strip())
            except Exception:
                lbl = -1
            label_counts[lbl] += 1

            split_clean = raw_split.strip().lower()
            split_counts[split_clean] += 1
            partition_label_matrix[(split_clean, lbl)] += 1

            if stmt:
                statement_types[stmt] += 1
            if tech:
                attack_techniques[tech] += 1
            if stage:
                attack_stages[stage] += 1
            if status:
                attack_statuses[status] += 1
            if tamper:
                tamper_methods[tamper] += 1
            if tmpl:
                unique_templates.add(tmpl)
            if atk_id:
                unique_attacks.add(atk_id)

            # Deduplication
            if full_query:
                if full_query in raw_queries_seen:
                    exact_duplicates += 1
                else:
                    raw_queries_seen.add(full_query)

                norm_q = normalize_sql(full_query)
                if norm_q in norm_queries_seen:
                    norm_duplicates += 1
                else:
                    norm_queries_seen.add(norm_q)

                if full_query not in query_to_labels:
                    query_to_labels[full_query] = set()
                query_to_labels[full_query].add(lbl)

            if max_rows and total_rows >= max_rows:
                break

    # Contradictory queries
    contradictory_queries = sum(1 for q, lbls in query_to_labels.items() if len(lbls) > 1)

    elapsed = time.time() - start_time
    logger.info(f"Audited {file_path.name}: {total_rows:,} rows in {elapsed:.2f}s.")

    return {
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "total_rows": total_rows,
        "header_columns": header,
        "missing_required_columns": [c for c in REQUIRED_COLUMNS if c not in col_to_idx],
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "partition_matrix": {f"split={s}_label={l}": count for (s, l), count in partition_label_matrix.items()},
        "statement_types": dict(statement_types),
        "attack_techniques": dict(attack_techniques),
        "attack_stages": dict(attack_stages),
        "attack_statuses": dict(attack_statuses),
        "unique_templates_count": len(unique_templates),
        "unique_attacks_count": len(unique_attacks),
        "unique_raw_queries": len(raw_queries_seen),
        "exact_duplicates_count": exact_duplicates,
        "unique_normalized_queries": len(norm_queries_seen),
        "normalized_duplicates_count": norm_duplicates,
        "contradictory_queries_count": contradictory_queries,
        "missing_counts_per_column": dict(missing_counts),
        "audit_duration_seconds": round(elapsed, 2),
    }


def generate_markdown_audit_report(audits: list[dict[str, Any]], output_path: Path) -> Path:
    """Generate comprehensive Markdown audit report from audit results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Real-Data Audit & Split-Provenance Confounding Report",
        "",
        "**Document Version:** 1.0  ",
        "**Authoritative Source:** Superviz26-SQL / Adapt-SQL (Zenodo DOI: [10.5281/zenodo.19627322](https://doi.org/10.5281/zenodo.19627322))  ",
        "",
        "---",
        "",
        "## 1. Summary of Downloaded Files",
        "",
        "| File Name | File Size | Total Rows | Unique Queries | Normal ($y=0$) | Attack ($y=1$) | Attack % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for a in audits:
        f_name = a["file_name"]
        f_size = f"{a['file_size_bytes'] / (1024*1024):.1f} MB"
        t_rows = f"{a['total_rows']:,}"
        u_q = f"{a['unique_raw_queries']:,}"
        n_0 = f"{a['label_counts'].get(0, 0):,}"
        n_1 = f"{a['label_counts'].get(1, 0):,}"
        pct = round(a["label_counts"].get(1, 0) / max(a["total_rows"], 1) * 100, 2)
        lines.append(f"| `{f_name}` | {f_size} | {t_rows} | {u_q} | {n_0} | {n_1} | {pct}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Partition x Label Breakdown (Split-Provenance Confounding Check)",
        "",
        "This table evaluates whether label is confounded with the official `split` column:",
        "",
        "| File Name | Split='train' Normal | Split='train' Attack | Split='test' Normal | Split='test' Attack | Confounding Severity |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    for a in audits:
        f_name = a["file_name"]
        pm = a["partition_matrix"]
        tr_0 = pm.get("split=train_label=0", 0)
        tr_1 = pm.get("split=train_label=1", 0)
        te_0 = pm.get("split=test_label=0", 0)
        te_1 = pm.get("split=test_label=1", 0)

        conf = "CRITICAL (0 attacks in train)" if tr_1 == 0 else "Balanced"
        lines.append(f"| `{f_name}` | {tr_0:,} | {tr_1:,} | {te_0:,} | {te_1:,} | **{conf}** |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Query Duplication & Collision Audit",
        "",
        "| File Name | Exact Duplicate Queries | Normalized Collisions | Contradictory Queries (Both 0 & 1) |",
        "| :--- | :---: | :---: | :---: |",
    ])

    for a in audits:
        lines.append(
            f"| `{a['file_name']}` | {a['exact_duplicates_count']:,} | "
            f"{a['normalized_duplicates_count']:,} | {a['contradictory_queries_count']:,} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Attack Technique Distribution",
        "",
    ])

    for a in audits:
        lines.append(f"### File: `{a['file_name']}`")
        lines.append("| Attack Technique | Query Count | % of Malicious |")
        lines.append("| :--- | :---: | :---: |")
        total_atk = a["label_counts"].get(1, 0)
        for tech, count in sorted(a["attack_techniques"].items(), key=lambda x: -x[1]):
            pct = round(count / max(total_atk, 1) * 100, 2)
            lines.append(f"| `{tech}` | {count:,} | {pct}% |")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved Markdown audit report to {output_path}")
    return output_path


if __name__ == "__main__":
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in data/")
        sys.exit(1)

    audits = [audit_csv_file(f) for f in csv_files]
    json_path = Path("results/audit/real_data_audit.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)

    md_path = Path("results/audit/real_data_audit.md")
    generate_markdown_audit_report(audits, md_path)
