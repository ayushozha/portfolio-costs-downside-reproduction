"""New descriptive comparison of retained cost-aware allocations to initialization.

Uses the executed frozen constructor and historical numerical runtime. No model
training, new performance metric, resampling, or causal decomposition is run.
"""
from __future__ import division

import argparse
import csv
import datetime
import json
import os
import platform
import sys

import numpy as np

import portfolio_engine_frozen as engine


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def compare(run_directory, protocol_path, output_directory):
    started = datetime.datetime.utcnow().isoformat()+"Z"
    if platform.python_version() != "3.6.2" or np.__version__ != "1.13.1":
        raise AssertionError("Use Python 3.6.2 / NumPy 1.13.1")
    if os.path.exists(output_directory):
        raise AssertionError("Refusing to overwrite an existing diagnostic execution")
    os.makedirs(output_directory)
    inputs = {}

    def record(path):
        inputs[path.replace("\\", "/")] = engine.sha256(path)
        return path

    original_execution = read_json(record(os.path.join(run_directory, "execution.json")))
    protocol = read_json(record(protocol_path))
    frozen_engine_path = os.path.relpath(engine.__file__)
    if engine.sha256(record(frozen_engine_path)) != original_execution["engine_sha256"]:
        raise AssertionError("Engine does not match the executed frozen engine")
    if engine.sha256(protocol_path) != original_execution["protocol_sha256"]:
        raise AssertionError("Protocol does not match the original run")
    variants = ["return_only", "cost_aware"]+[
        "downside_lambda{}".format(value) for value in protocol["downside_lambdas"]]
    selections, rows = [], []
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        folder = os.path.join(run_directory, dataset_id)
        data_record = read_json(record(os.path.join(folder, "data_record.json")))
        if engine.sha256(record(dataset["path"])) != data_record["data_sha256"]:
            raise AssertionError("Dataset differs from retained input")
        risky = engine.read_relatives(dataset["path"])
        states, unused_relatives, indices = engine.make_samples(risky, protocol["lookback"])
        with np.load(record(os.path.join(folder, "training_standardization.npz"))) as standard:
            states = (states-standard["center"])/standard["scale"]
        for variant in variants:
            for seed in protocol["seeds"]:
                name = "{}_seed{}".format(variant, seed)
                fit = read_json(record(os.path.join(folder, name+"_training.json")))
                if fit["seed"] != seed or fit["config"]["name"] != variant:
                    raise AssertionError("Fit identity mismatch")
                selected_path = record(os.path.join(folder, name+".npz"))
                if engine.sha256(selected_path) != fit["selected_parameters_sha256"]:
                    raise AssertionError("Selected parameters differ from retained fit")
                initial = engine.Policy(states.shape[1], risky.shape[1]+1,
                                        protocol["hidden_sizes"], seed)
                selected = engine.Policy(states.shape[1], risky.shape[1]+1,
                                         protocol["hidden_sizes"], seed)
                with np.load(selected_path) as saved:
                    selected.parameters = [saved["parameter_{}".format(i)].copy()
                                           for i in range(len(initial.parameters))]
                if any(left.shape != right.shape for left, right in
                       zip(initial.parameters, selected.parameters)):
                    raise AssertionError("Parameter shape mismatch")
                identical = all(np.array_equal(left, right) for left, right in
                                zip(initial.parameters, selected.parameters))
                if fit["selected_update"] == 0 and not identical:
                    raise AssertionError("Update-zero parameters do not match reconstructed initialization")
                selections.append({"dataset": dataset_id, "policy": variant,
                                   "seed": seed, "selected_update": fit["selected_update"],
                                   "selected_parameters_exactly_initial": identical})
                if variant != "cost_aware":
                    continue
                if fit["config"]["train_cost"] != protocol["train_cost"] or \
                        fit["config"]["downside_lambda"] != 0:
                    raise AssertionError("Cost-aware configuration mismatch")
                for partition in ["train", "validation", "test"]:
                    mask = (indices >= dataset[partition][0]) & (indices < dataset[partition][1])
                    initial_weights, unused = initial.forward(states[mask])
                    selected_weights, unused = selected.forward(states[mask])
                    difference = np.abs(selected_weights-initial_weights)
                    l1 = np.sum(difference, axis=1)
                    if identical and np.any(difference != 0):
                        raise AssertionError("Identical parameters produce different allocations")
                    verified_trajectory_error = None
                    if partition == "test":
                        trajectory_path = record(os.path.join(folder, name+"_cost{}.csv".format(
                            protocol["primary_evaluation_cost"])))
                        trajectory = np.loadtxt(trajectory_path, delimiter=",", skiprows=1)
                        if not np.array_equal(trajectory[:, 0], indices[mask]):
                            raise AssertionError("Retained trajectory has different observations")
                        verified_trajectory_error = float(np.max(np.abs(
                            selected_weights-trajectory[:, 6:])))
                        if verified_trajectory_error > 1e-14:
                            raise AssertionError("Selected allocations do not match retained trajectory")
                    rows.append({"dataset": dataset_id, "policy": variant, "seed": seed,
                                 "partition": partition, "observations": int(np.sum(mask)),
                                 "selected_update": fit["selected_update"],
                                 "selected_parameters_exactly_initial": identical,
                                 "allocation_l1_mean": float(np.mean(l1)),
                                 "allocation_l1_maximum": float(np.max(l1)),
                                 "allocation_component_absolute_difference_maximum": float(np.max(difference)),
                                 "retained_test_allocation_absolute_error_maximum": verified_trajectory_error})
    if len(selections) != 75 or len(rows) != 45:
        raise AssertionError("Unexpected fit or diagnostic row count")
    counts = []
    for dataset in protocol["datasets"]:
        for variant in variants:
            matches = [row for row in selections if row["dataset"] == dataset["id"] and row["policy"] == variant]
            counts.append({"dataset": dataset["id"], "policy": variant, "fits": len(matches),
                           "update_zero_count": sum(row["selected_update"] == 0 for row in matches),
                           "selected_updates": [row["selected_update"] for row in matches]})
    summary = []
    for dataset in protocol["datasets"]:
        matches = [row for row in rows if row["dataset"] == dataset["id"] and row["partition"] == "test"]
        nonzero = [row for row in matches if row["selected_update"] != 0]
        summary.append({"dataset": dataset["id"], "fits": len(matches),
                        "update_zero_count": sum(row["selected_update"] == 0 for row in matches),
                        "test_mean_l1_per_seed": [row["allocation_l1_mean"] for row in matches],
                        "test_mean_l1_seed_median": float(np.median([row["allocation_l1_mean"] for row in matches])),
                        "test_mean_l1_seed_minimum": float(min(row["allocation_l1_mean"] for row in matches)),
                        "test_mean_l1_seed_maximum": float(max(row["allocation_l1_mean"] for row in matches)),
                        "nonzero_update_test_mean_l1_per_seed": [row["allocation_l1_mean"] for row in nonzero],
                        "test_l1_maximum_over_all_seeds_and_observations": float(max(row["allocation_l1_maximum"] for row in matches)),
                        "test_component_difference_maximum_over_all_seeds_and_observations": float(max(row["allocation_component_absolute_difference_maximum"] for row in matches))})
    result = {"status": "PASS", "scope": "Post hoc descriptive allocations only; cost-aware fits compared with same-seed reconstructed initialization on fixed train/validation/test states. All 75 retained fit records and parameter identities checked; 15 cost-aware fits evaluated.",
              "allocation_l1_definition": "sum over cash and all risky components of abs(selected_weight-initial_weight), for each observation; lies in [0,2] for simplex allocations",
              "maximum_component_definition": "maximum abs(selected_weight-initial_weight) across all observations and allocation components",
              "initialization_recovery": "Executed frozen Policy constructor, same input/output dimensions, hidden sizes and NumPy RandomState seed; original standardization retained",
              "rows": rows, "cost_aware_test_summary": summary, "checkpoint_counts": counts,
              "all_fit_selections": selections,
              "all_fit_count": len(selections), "all_update_zero_count": sum(row["selected_update"] == 0 for row in selections),
              "cost_aware_fit_count": 15, "cost_aware_update_zero_count": sum(row["selected_update"] == 0 for row in selections if row["policy"] == "cost_aware"),
              "zero_checkpoint_parameters_exactly_initial_count": sum(row["selected_update"] == 0 and row["selected_parameters_exactly_initial"] for row in selections),
              "limitations": ["Post hoc descriptive diagnostic, not a prespecified inferential test", "No threshold for near initialization was prespecified", "Does not identify causes of a performance difference", "No new training or checkpoint selection", "Initial parameter arrays reconstructed from the recorded generator rather than separately saved initial checkpoint files", "All recoverable update-zero parameter arrays match exactly; test allocations independently match retained paths"],
              "no_training": True, "no_performance_recalculation": True, "no_bootstrap_or_causal_decomposition": True}
    csv_path = os.path.join(output_directory, "allocation_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(os.path.join(output_directory, "comparison.json"), result)
    completed = datetime.datetime.utcnow().isoformat()+"Z"
    # Recheck every consumed input so this fresh computation cannot silently
    # inherit an execution timestamp or a modified artifact from the old run.
    if any(engine.sha256(path) != value for path, value in inputs.items()):
        raise AssertionError("An input changed during the descriptive execution")
    write_json(os.path.join(output_directory, "execution.json"), {
        "actual_start_utc": started, "actual_completed_utc": completed,
        "original_training_start_utc": original_execution["actual_start_utc"],
        "original_training_completed_utc": original_execution["actual_completed_utc"],
        "this_is_a_new_diagnostic_execution": True, "fictional_historical_execution": False,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "platform": platform.platform(), "arguments": sys.argv,
        "script_sha256": engine.sha256(__file__), "input_sha256": inputs,
        "output_sha256": {"allocation_comparison.csv": engine.sha256(csv_path),
                          "comparison.json": engine.sha256(os.path.join(output_directory, "comparison.json"))},
        "no_training": True, "no_bootstrap": True, "no_performance_recalculation": True})
    print(json.dumps({"status": "PASS", "all_fit_count": 75,
                      "all_update_zero_count": result["all_update_zero_count"],
                      "cost_aware_update_zero_count": result["cost_aware_update_zero_count"],
                      "cost_aware_test_summary": summary, "output": output_directory}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", default="evidence/runs/main-v2")
    parser.add_argument("--protocol", default="experiment/protocol.json")
    parser.add_argument("--output", default="evidence/analysis/initialization-comparison")
    args = parser.parse_args()
    compare(args.run_directory, args.protocol, args.output)
