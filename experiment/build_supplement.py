"""Format recorded primary metrics; compatible with the historical runtime.

No experiment or inference is changed. The internal saved downside-ratio field
is presented under its explicit, neutral zero-target metric name.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform

import numpy as np


MAPPING = [
    ("annualized_return", "annualized_return", "Annualized return (percent)", 100.0, 3),
    ("annualized_sharpe_zero_cash_rate", "annualized_sharpe_zero_cash_rate", "Zero-cash-rate annualized Sharpe", 1.0, 3),
    ("zero_target_downside_ratio", "annualized_sortino_zero_target", "Zero-target downside ratio", 1.0, 3),
    ("total_log_fee_drag", "total_log_fee_drag", "Total log fee drag", 1.0, 6),
]


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def display_value(value, factor, digits):
    if value == "":
        return "undefined"
    number = float(value)*factor
    if not np.isfinite(number):
        raise AssertionError("A saved metric is non-finite")
    return ("{:."+str(digits)+"f}").format(number)


def metric_cell(row, source_name, factor, digits):
    median = display_value(row[source_name+"_median"], factor, digits)
    if row["seed_count"] == "1":
        return median
    return "{} [{}, {}]".format(
        median, display_value(row[source_name+"_minimum"], factor, digits),
        display_value(row[source_name+"_maximum"], factor, digits))


def generate(source, manuscript, supplemental_csv, verification, replace_generated=False):
    started = datetime.datetime.utcnow().isoformat()+"Z"
    if platform.python_version() != "3.6.2" or np.__version__ != "1.13.1":
        raise AssertionError("Use Python 3.6.2 / NumPy 1.13.1 for the recorded scientific table")
    for output in [manuscript, supplemental_csv, verification]:
        if os.path.exists(output) and not replace_generated:
            raise AssertionError("Refusing to replace existing supplemental artifact: "+output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(source, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24 or len(set((row["dataset"], row["policy"]) for row in rows)) != 24:
        raise AssertionError("Expected exactly 24 distinct primary rows")
    if any(float(row["evaluation_cost"]) != 0.0025 for row in rows):
        raise AssertionError("Supplement must contain only the prespecified primary fee")
    public_rows = []
    for row in rows:
        public = dict((name, row[name]) for name in ["dataset", "policy", "evaluation_cost", "observations", "seed_count", "summary_type"])
        for public_name, source_name, unused_label, unused_factor, unused_digits in MAPPING:
            for suffix in ["median", "minimum", "maximum", "valid_count", "undefined_count"]:
                public[public_name+"_"+suffix] = row[source_name+"_"+suffix]
        public_rows.append(public)
    with open(supplemental_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(public_rows[0].keys()))
        writer.writeheader()
        writer.writerows(public_rows)
    lines = ["# Appendix A. Additional performance measures", "",
             "Tables A1-A3 report additional descriptive measures at the primary transaction-cost coefficient of 0.0025 per risky dollar traded. The definitions and conventional annualization at 252 observations per year are those in Section III.E. Annualized return uses net log growth; the Sharpe statistic uses net simple returns and zero cash yield. The zero-target downside ratio uses mean net simple return relative to a zero target divided by daily downside deviation, with the annualization factor specified in the methods. Total log fee drag is the accumulated negative logarithm of the fee multiplier, rather than a percentage of initial wealth. [claim:res_supplement]", "",
             "Each neural-policy entry is the median [minimum, maximum] across all five seeds. Deterministic baselines have one value. Ratios with zero denominators remain undefined. Display rounding does not change the saved metrics, and annualization remains descriptive rather than a forecast. [claim:res_supplement]", ""]
    labels = {"nyse-o": "NYSE(O)", "sp500": "SP500", "djia": "DJIA"}
    policy_labels = {"cash": "Cash", "equal_weight_crp": "Equal-weight rebalancing", "equal_weight_buy_hold": "Buy-and-hold",
                     "return_only": "Return-only", "cost_aware": "Cost-aware", "downside_lambda1": "Downside lambda 1",
                     "downside_lambda10": "Downside lambda 10", "downside_lambda100": "Downside lambda 100"}
    rendered_rows = []
    for table_index, dataset in enumerate(["nyse-o", "sp500", "djia"], 1):
        table_number = "A{}".format(table_index)
        lines.extend(["## Table "+table_number+": "+labels[dataset], "",
                      "| Policy | "+" | ".join(item[2] for item in MAPPING)+" |",
                      "|---|---:|---:|---:|---:|"])
        for row in [item for item in rows if item["dataset"] == dataset]:
            cells = [policy_labels[row["policy"]]]+[
                metric_cell(row, source_name, factor, digits)
                for unused_public, source_name, unused_label, factor, digits in MAPPING]
            rendered_rows.append({"dataset": dataset, "policy": row["policy"], "cells": cells[1:]})
            lines.append("| "+" | ".join(cells)+" |")
        lines.extend(["", "Table "+table_number+". "+labels[dataset]+" primary-fee descriptive metrics. [claim:res_supplement]", ""])
    with open(manuscript, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    # Transfer verification is independent of the formatting loop: reopen both
    # CSV files, preserve source strings, then account for every displayed cell.
    with open(source, encoding="utf-8") as handle:
        saved = dict(((row["dataset"], row["policy"]), row) for row in csv.DictReader(handle))
    with open(supplemental_csv, encoding="utf-8") as handle:
        transferred = list(csv.DictReader(handle))
    exact_numeric_cells = 0
    exact_undefined_cells = 0
    for row in transferred:
        original = saved[(row["dataset"], row["policy"])]
        for public_name, source_name, unused_label, unused_factor, unused_digits in MAPPING:
            for suffix in ["median", "minimum", "maximum"]:
                if row[public_name+"_"+suffix] != original[source_name+"_"+suffix]:
                    raise AssertionError("Supplemental CSV differs from original saved cell")
                if row[public_name+"_"+suffix] == "":
                    exact_undefined_cells += 1
                else:
                    exact_numeric_cells += 1
    if exact_numeric_cells+exact_undefined_cells != 288:
        raise AssertionError("Unexpected number of supplemental summary cells")
    with open(manuscript, encoding="utf-8") as handle:
        markdown_rows = [line.strip().strip("|").split("|") for line in handle
                         if line.startswith("| ") and not line.startswith("| Policy")]
    if len(markdown_rows) != 24:
        raise AssertionError("Markdown omits or duplicates a supplemental row")
    for cells, identity in zip(markdown_rows, rendered_rows):
        original = saved[(identity["dataset"], identity["policy"])]
        for column, mapping in enumerate(MAPPING):
            source_name, factor, digits = mapping[1], mapping[3], mapping[4]
            values = []
            for suffix in ["median", "minimum", "maximum"]:
                raw = original[source_name+"_"+suffix]
                values.append("undefined" if raw == "" else format(float(raw)*factor, ".{}f".format(digits)))
            expected_cell = values[0] if original["seed_count"] == "1" else \
                values[0]+" ["+values[1]+", "+values[2]+"]"
            if cells[column+1].strip() != expected_cell:
                raise AssertionError("Rendered metric cell differs from saved source")
    report = {"status": "PASS", "verification_scope": "Every unrounded median/minimum/maximum CSV cell equals the saved primary summary string; every rendered cell records its source identity and display convention. No metrics, models or inference recomputed.",
              "source_path": source, "source_sha256": sha256(source),
              "supplemental_csv_path": supplemental_csv, "supplemental_csv_sha256": sha256(supplemental_csv),
              "manuscript_path": manuscript, "manuscript_sha256": sha256(manuscript),
              "generator_sha256": sha256(__file__), "rows": len(transferred),
              "exact_numeric_summary_cells": exact_numeric_cells, "preserved_undefined_summary_cells": exact_undefined_cells,
              "rendered_metric_cells": 96, "rendered_metric_cells_verified": 96,
              "display_mapping": MAPPING, "rendered_rows": rendered_rows,
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "actual_start_utc": started, "actual_completed_utc": datetime.datetime.utcnow().isoformat()+"Z",
              "table_formatting_only": True, "no_model_training": True, "no_inference_change": True}
    with open(verification, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"status": "PASS", "rows": len(transferred), "rendered_metric_cells": 96,
                      "exact_numeric_summary_cells": exact_numeric_cells, "undefined_summary_cells": exact_undefined_cells,
                      "manuscript": manuscript, "supplemental_csv": supplemental_csv}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="evidence/analysis/main-v2/primary_summary.csv")
    parser.add_argument("--manuscript", default="manuscript/supplement.md")
    parser.add_argument("--csv", default="evidence/analysis/main-v2/supplement_metrics.csv")
    parser.add_argument("--verification", default="evidence/analysis/main-v2/supplement_verification.json")
    parser.add_argument("--replace-generated", action="store_true",
                        help="Refresh the generated table artifacts; original analysis inputs remain read-only")
    args = parser.parse_args()
    generate(args.source, args.manuscript, args.csv, args.verification, args.replace_generated)
