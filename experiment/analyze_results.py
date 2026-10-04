"""Frozen-protocol summaries and stationary bootstrap in the historical runtime.

Runs with Python 3.6.2 and NumPy 1.13.1. It consumes recorded trajectories;
it does not train, choose a model, alter the protocol, or replace a run.
"""
from __future__ import division

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np


METRICS = ["terminal_wealth", "cumulative_return", "mean_daily_log_return",
           "annualized_return", "annualized_volatility",
           "annualized_downside_deviation", "annualized_sharpe_zero_cash_rate",
           "annualized_sortino_zero_target", "maximum_drawdown",
           "mean_risky_dollar_turnover", "total_log_fee_drag", "negative_return_fraction",
           "mean_daily_cash_weight", "last_cash_weight"]


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = sorted(set(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {"median": None, "minimum": None, "maximum": None,
                "valid_count": 0, "undefined_count": len(values)}
    if not np.isfinite(valid).all():
        raise AssertionError("Non-finite metric in saved run")
    return {"median": float(np.median(valid)), "minimum": float(np.min(valid)),
            "maximum": float(np.max(valid)), "valid_count": len(valid),
            "undefined_count": len(values)-len(valid)}


def stationary_indices(observations, replicates, mean_block_length, seed):
    """Politis-Romano: geometric restart, uniform start, circular continuation."""
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, observations, size=(replicates, observations))
    restart = rng.uniform(size=(replicates, observations)) < 1.0/mean_block_length
    indices = np.empty((replicates, observations), dtype=np.int64)
    indices[:, 0] = starts[:, 0]
    for column in range(1, observations):
        indices[:, column] = np.where(restart[:, column], starts[:, column],
                                      (indices[:, column-1]+1) % observations)
    return indices


def format_value(value, decimals=6):
    return "undefined" if value is None else ("{:."+str(decimals)+"f}").format(value)


def analyze(protocol_path, results_path, output_path):
    started = datetime.datetime.utcnow().isoformat()+"Z"
    protocol = read_json(protocol_path)
    expected_runtime = protocol["scientific_runtime"]
    if platform.python_version() != expected_runtime["python_version"] or \
            np.__version__ != expected_runtime["numpy_version"]:
        raise AssertionError("Analysis must use the frozen historical scientific runtime")
    execution_path = os.path.join(results_path, "execution.json")
    execution = read_json(execution_path)
    if execution["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Protocol changed after the recorded run")
    if execution["python_version"] != expected_runtime["python_version"] or \
            execution["numpy_version"] != expected_runtime["numpy_version"]:
        raise AssertionError("The recorded experiment used an unexpected runtime")
    if os.path.exists(output_path):
        raise AssertionError("Refusing to replace an existing analysis directory")
    os.makedirs(output_path)
    input_records = []

    def record_input(path, role):
        input_records.append({"path": os.path.normpath(path).replace("\\", "/"),
                              "role": role, "sha256": digest(path)})

    record_input(protocol_path, "frozen_protocol")
    record_input(execution_path, "experiment_execution")
    metrics_path = os.path.join(results_path, "metrics.json")
    record_input(metrics_path, "saved_metrics")
    rows = read_json(metrics_path)
    expected_rows = len(protocol["datasets"])*len(protocol["evaluation_costs"])*(
        len(protocol["baselines"])+(2+len(protocol["downside_lambdas"]))*len(protocol["seeds"]))
    if len(rows) != expected_rows or execution["scientific_result_rows"] != expected_rows:
        raise AssertionError("Missing saved metric rows")
    expected_policies = protocol["baselines"]+["return_only", "cost_aware"]+[
        "downside_lambda{}".format(value) for value in protocol["downside_lambdas"]]
    seen = set()
    trajectories = {}
    checkpoints = []
    primary_daily_rows = []
    groups = {}
    for row in rows:
        key = (row["dataset"], row["policy"], row["seed"], row["evaluation_cost"])
        if key in seen:
            raise AssertionError("Duplicate metric identity")
        seen.add(key)
        if row["policy"] not in expected_policies or row["evaluation_cost"] not in protocol["evaluation_costs"]:
            raise AssertionError("Undeclared policy or fee")
        name = row["policy"] if row["seed"] is None else "{}_seed{}".format(row["policy"], row["seed"])
        directory = os.path.join(results_path, row["dataset"])
        path = os.path.join(directory, "{}_cost{}.csv".format(name, row["evaluation_cost"]))
        record_input(path, "saved_daily_trajectory")
        with open(path, encoding="utf-8") as stream:
            table = list(csv.DictReader(stream))
        series = dict((field, np.array([float(record[field]) for record in table]))
                      for field in ["net_log_return", "cash_weight"])
        indices = np.array([int(record["source_return_index_zero_based"]) for record in table])
        if len(table) != row["observations"] or not np.isfinite(series["net_log_return"]).all() \
                or not np.isfinite(series["cash_weight"]).all():
            raise AssertionError("Invalid trajectory length or numbers")
        if abs(float(np.mean(series["net_log_return"]))-row["mean_daily_log_return"]) > 2e-12:
            raise AssertionError("Saved log-return metric differs from the trajectory")
        if np.min(series["cash_weight"]) < 0 or np.max(series["cash_weight"]) > 1:
            raise AssertionError("Cash allocation outside simplex")
        row["mean_daily_cash_weight"] = float(np.mean(series["cash_weight"]))
        row["last_cash_weight"] = float(series["cash_weight"][-1])
        trajectories[key] = {"log_returns": series["net_log_return"],
                             "cash_weight": series["cash_weight"], "source_indices": indices}
        groups.setdefault((row["dataset"], row["policy"], row["evaluation_cost"]), []).append(row)

    summaries = []
    checkpoint_summary = []
    paired = []
    bootstrap = []
    primary_cost = protocol["primary_evaluation_cost"]
    analysis = protocol["analysis"]
    summary_map = {}
    seed_list = protocol["seeds"]
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        data_path = os.path.join(results_path, dataset_id, "data_record.json")
        record_input(data_path, "dataset_run_record")
        data_record = read_json(data_path)
        if data_record["data_sha256"] != digest(dataset["path"]):
            raise AssertionError("Dataset CSV differs from the run record")
        record_input(dataset["path"], "historical_dataset_csv")
        expected_indices = np.arange(dataset["test"][0], dataset["test"][1])
        block_indices = stationary_indices(len(expected_indices), analysis["stationary_bootstrap_replicates"],
                                           analysis["stationary_bootstrap_mean_block_length"],
                                           analysis["bootstrap_seed"])
        for policy in expected_policies:
            seeds = [None] if policy in protocol["baselines"] else seed_list
            for cost in protocol["evaluation_costs"]:
                group = groups[(dataset_id, policy, cost)]
                if set(row["seed"] for row in group) != set(seeds) or len(group) != len(seeds):
                    raise AssertionError("Missing or repeated expected seeds")
                summary = {"dataset": dataset_id, "policy": policy, "evaluation_cost": cost,
                           "evaluation_cost_basis_points": cost*10000,
                           "observations": len(expected_indices), "seed_count": len(seeds),
                           "summary_type": "deterministic_baseline" if seeds == [None] else "seed_median_range"}
                for metric in METRICS:
                    statistic = describe([row[metric] for row in group])
                    for suffix, value in statistic.items():
                        summary[metric+"_"+suffix] = value
                summaries.append(summary)
                summary_map[(dataset_id, policy, cost)] = summary
                for seed in seeds:
                    if not np.array_equal(trajectories[(dataset_id, policy, seed, cost)]["source_indices"], expected_indices):
                        raise AssertionError("Incorrect chronological trajectory indices")
            primary_trajectories = [trajectories[(dataset_id, policy, seed, primary_cost)] for seed in seeds]
            cash = np.array([item["cash_weight"] for item in primary_trajectories])
            logs = np.array([item["log_returns"] for item in primary_trajectories])
            for position, index in enumerate(expected_indices):
                primary_daily_rows.append({"dataset": dataset_id, "policy": policy, "source_return_index_zero_based": int(index),
                                           "evaluation_cost": primary_cost, "seed_count": len(seeds),
                                           "cash_weight_seed_mean": float(np.mean(cash[:, position])),
                                           "cash_weight_seed_minimum": float(np.min(cash[:, position])),
                                           "cash_weight_seed_maximum": float(np.max(cash[:, position])),
                                           "net_log_return_seed_mean": float(np.mean(logs[:, position]))})
            if seeds != [None]:
                selected = []
                for seed in seeds:
                    training_path = os.path.join(results_path, dataset_id, "{}_seed{}_training.json".format(policy, seed))
                    record_input(training_path, "saved_training_selection")
                    training = read_json(training_path)
                    if training["selected_update"] != max(training["history"], key=lambda item: item["validation_objective"])["update"]:
                        raise AssertionError("Selected checkpoint differs from validation criterion")
                    selected.append(training["selected_update"])
                    checkpoints.append({"dataset": dataset_id, "policy": policy, "seed": seed,
                                        "selected_update": training["selected_update"],
                                        "checkpoint_zero_selected": training["selected_update"] == 0,
                                        "selected_validation_objective": training["selected_validation_objective"],
                                        "recorded_candidate_checkpoints": len(training["history"])})
                checkpoint_summary.append({"dataset": dataset_id, "policy": policy,
                                           "seeds": len(seeds), "checkpoint_zero_count": sum(value == 0 for value in selected),
                                           "selected_updates_by_seed": selected})
        for cost in protocol["evaluation_costs"]:
            for left, right in analysis["primary_paired_contrasts"]:
                label = "{}_minus_{}".format(left, right)
                group_summary = {"dataset": dataset_id, "contrast": label, "left_policy": left,
                                 "right_policy": right, "evaluation_cost": cost,
                                 "primary": cost == primary_cost, "seed_count": len(seed_list)}
                for metric in METRICS:
                    differences = []
                    for seed in seed_list:
                        left_row = next(row for row in groups[(dataset_id, left, cost)] if row["seed"] == seed)
                        right_row = next(row for row in groups[(dataset_id, right, cost)] if row["seed"] == seed)
                        differences.append(None if left_row[metric] is None or right_row[metric] is None
                                           else left_row[metric]-right_row[metric])
                    group_summary[metric] = dict(describe(differences), values_by_seed=differences)
                paired.append(group_summary)
                daily_difference = np.mean(np.array([
                    trajectories[(dataset_id, left, seed, cost)]["log_returns"]-
                    trajectories[(dataset_id, right, seed, cost)]["log_returns"] for seed in seed_list]), axis=0)
                sample_means = np.mean(daily_difference[block_indices], axis=1)
                low, high = np.percentile(sample_means, [2.5, 97.5])
                point = float(np.mean(daily_difference))
                bootstrap.append({"dataset": dataset_id, "contrast": label, "left_policy": left, "right_policy": right,
                                  "evaluation_cost": cost, "primary": cost == primary_cost,
                                  "observations": len(expected_indices), "seeds_averaged": len(seed_list),
                                  "mean_daily_log_return_difference": point,
                                  "interval_lower": float(low), "interval_upper": float(high),
                                  "mean_daily_difference_basis_points": point*10000,
                                  "interval_lower_basis_points": float(low)*10000,
                                  "interval_upper_basis_points": float(high)*10000,
                                  "interval_contains_zero": bool(low <= 0 <= high),
                                  "interval_method": "stationary_bootstrap_percentile",
                                  "interval_level": analysis["interval_level"],
                                  "bootstrap_replicates": analysis["stationary_bootstrap_replicates"],
                                  "mean_block_length": analysis["stationary_bootstrap_mean_block_length"],
                                  "bootstrap_seed": analysis["bootstrap_seed"],
                                  "inference_scope": analysis["inference_scope"]})
                np.save(os.path.join(output_path, "{}_{}_cost{}_bootstrap_means.npy".format(dataset_id, label, cost)), sample_means)
    primary = [row for row in summaries if row["evaluation_cost"] == primary_cost]
    write_csv(os.path.join(output_path, "seed_metrics.csv"), rows)
    write_csv(os.path.join(output_path, "primary_summary.csv"), primary)
    write_csv(os.path.join(output_path, "cost_sweep_summary.csv"), summaries)
    write_csv(os.path.join(output_path, "checkpoint_selection.csv"), checkpoints)
    write_csv(os.path.join(output_path, "daily_primary_averages.csv"), primary_daily_rows)
    write_csv(os.path.join(output_path, "bootstrap_contrasts.csv"), bootstrap)
    checkpoint_zero_count = sum(row["checkpoint_zero_selected"] for row in checkpoints)
    report = {"source_run_directory": os.path.normpath(results_path).replace("\\", "/"),
              "scientific_runtime": expected_runtime, "metric_rows_preserved": len(rows),
              "summary_rows": len(summaries), "primary_summary_rows": len(primary),
              "checkpoint_records": len(checkpoints), "checkpoint_zero_selected": checkpoint_zero_count,
              "checkpoint_summary": checkpoint_summary, "seed_order": seed_list,
              "primary_evaluation_cost": primary_cost, "primary_contrast_count": sum(row["primary"] for row in bootstrap),
              "sensitivity_contrast_count": sum(not row["primary"] for row in bootstrap),
              "summaries": summaries, "paired_seed_differences": paired, "bootstrap_contrasts": bootstrap,
              "scope": "Descriptive seed median/min/max and exploratory stationary-bootstrap percentile intervals on seed-averaged paired daily log-return differences, conditional on fitted models; no multiple-testing correction; no training uncertainty bootstrap",
              "source_ids": ["politis1994", "python362", "numpy1131"],
              "protocol_sha256": digest(protocol_path), "analysis_code_sha256": digest(__file__),
              "actual_start_utc": started, "actual_completed_utc": datetime.datetime.utcnow().isoformat()+"Z",
              "input_artifacts": input_records}
    write_json(os.path.join(output_path, "analysis.json"), report)
    execution_record = {"actual_start_utc": started, "actual_completed_utc": report["actual_completed_utc"],
                        "python_version": platform.python_version(), "numpy_version": np.__version__,
                        "platform": platform.platform(), "arguments": sys.argv,
                        "analysis_code_sha256": report["analysis_code_sha256"], "protocol_sha256": report["protocol_sha256"],
                        "source_run_execution_sha256": digest(execution_path),
                        "input_artifacts": input_records, "fictional_historical_execution": False}
    write_json(os.path.join(output_path, "execution.json"), execution_record)
    findings = ["# Recorded analysis findings", "", "Source run: `{}`. Actual analysis completion: {}.".format(results_path, report["actual_completed_utc"]),
                "", "All {} metric rows and {} trained policy/seed selections are retained. {} of these selections chose update zero. The zero-update checkpoint is a real randomly initialized network, not a successfully trained policy.".format(len(rows), len(checkpoints), checkpoint_zero_count),
                "", "Primary fee: {:.4f} per unit of risky dollar trade ({:.0f} basis points). Tables show median [minimum, maximum] across the saved seeds; deterministic baselines have a single value. Wealth starts from one, and terminal liquidation is absent.".format(primary_cost, primary_cost*10000), ""]
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        findings.extend(["## {}: primary-fee outcomes".format(dataset_id), "",
                         "| Policy | Terminal wealth | Maximum drawdown | Annualized downside deviation | Mean risky dollar turnover | Mean daily cash allocation |",
                         "|---|---:|---:|---:|---:|---:|"])
        for policy in expected_policies:
            row = summary_map[(dataset_id, policy, primary_cost)]
            cells = [policy]
            for metric in ["terminal_wealth", "maximum_drawdown", "annualized_downside_deviation", "mean_risky_dollar_turnover", "mean_daily_cash_weight"]:
                middle = format_value(row[metric+"_median"])
                cells.append(middle if row["seed_count"] == 1 else "{} [{}, {}]".format(middle, format_value(row[metric+"_minimum"]), format_value(row[metric+"_maximum"])))
            findings.append("| "+" | ".join(cells)+" |")
        findings.append("")
        for record in [item for item in bootstrap if item["dataset"] == dataset_id and item["primary"]]:
            findings.append("- {}: seed-averaged mean daily log-return difference {:.9f}, exploratory 95% stationary-bootstrap percentile interval [{:.9f}, {:.9f}]; zero is {}. This interval is conditional on the fitted models.".format(record["contrast"], record["mean_daily_log_return_difference"], record["interval_lower"], record["interval_upper"], "inside" if record["interval_contains_zero"] else "outside"))
        findings.append("")
    findings.extend(["## Interpretation limits", "", "The six primary contrasts use the prespecified primary fee; eighteen fee-sensitivity contrasts are identified separately. Bootstrap resampling uses circular geometric blocks with mean length 20, 2,000 replicates and seed 170724. The statistic averages paired daily log-return differences across the five fitted seeds before resampling time. It does not resample training, estimate independent seed populations, establish deployment profitability, or adjust for multiple comparisons. Small benchmark tails, dependence, retrospective constituents and omitted market frictions limit interpretation.",
                     "", "Cash allocations are observed actions, not an imposed finding. Compare downside changes with cash allocation and return changes before interpreting a penalty as improved investment performance. Undefined zero-volatility performance ratios remain null in the tables.", ""])
    with open(os.path.join(output_path, "findings.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(findings))
    print(json.dumps({"analysis": os.path.join(output_path, "analysis.json"),
                      "metric_rows": len(rows), "training_records": len(checkpoints),
                      "checkpoint_zero_selected": checkpoint_zero_count,
                      "primary_contrasts": report["primary_contrast_count"],
                      "sensitivity_contrasts": report["sensitivity_contrast_count"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.protocol, args.results, args.output)
