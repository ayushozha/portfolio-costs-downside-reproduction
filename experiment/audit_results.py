"""Independent accounting/output audit, compatible with the historical runtime.

No model selection or new strategy evaluation is performed by this verifier.
It reconstructs the arithmetic and artifacts already emitted by a frozen run.
"""
from __future__ import division

import argparse
import csv
import hashlib
import json
import os

import numpy as np


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def close(actual, expected, label, maximum_errors):
    error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    maximum_errors[label] = max(maximum_errors.get(label, 0.0), error)
    if error > 2e-10:
        raise AssertionError("{} mismatch {}".format(label, error))


def audit(protocol_path, result_directory):
    protocol = read_json(protocol_path)
    execution = read_json(os.path.join(result_directory, "execution.json"))
    if execution["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("The current protocol differs from the executed protocol")
    expected = protocol["scientific_runtime"]
    if execution["python_version"] != expected["python_version"] or \
            execution["numpy_version"] != expected["numpy_version"]:
        raise AssertionError("Executed software differs from the historical protocol")
    metric_rows = read_json(os.path.join(result_directory, "metrics.json"))
    maximum_errors = {}
    trajectories_checked = 0
    training_runs_checked = 0
    checkpoint_zero_selected = 0
    for dataset in protocol["datasets"]:
        source = np.loadtxt(dataset["path"], delimiter=",", skiprows=1)
        directory = os.path.join(result_directory, dataset["id"])
        record = read_json(os.path.join(directory, "data_record.json"))
        if record["data_sha256"] != digest(dataset["path"]):
            raise AssertionError("Source data changed after execution")
        history = np.log(source)
        train_indices = range(max(protocol["lookback"], dataset["train"][0]), dataset["train"][1])
        train_states = np.asarray([history[i-protocol["lookback"]:i].reshape(-1) for i in train_indices])
        norm = np.load(os.path.join(directory, "training_standardization.npz"))
        close(norm["center"], np.mean(train_states, axis=0), "training_only_center", maximum_errors)
        close(norm["scale"], np.maximum(np.std(train_states, axis=0), 1e-6),
              "training_only_scale", maximum_errors)
        test_indices = np.arange(dataset["test"][0], dataset["test"][1])
        future = np.column_stack((np.ones(len(test_indices)), source[test_indices]))
        test_states = np.asarray([history[i-protocol["lookback"]:i].reshape(-1) for i in test_indices])
        test_states = (test_states-norm["center"])/norm["scale"]
        policy_actions = {}
        for row in [r for r in metric_rows if r["dataset"] == dataset["id"]]:
            run = row["policy"] if row["seed"] is None else "{}_seed{}".format(row["policy"], row["seed"])
            csv_path = os.path.join(directory, "{}_cost{}.csv".format(run, row["evaluation_cost"]))
            with open(csv_path) as handle:
                table = list(csv.DictReader(handle))
            indices = np.array([int(record["source_return_index_zero_based"]) for record in table])
            if not np.array_equal(indices, test_indices):
                raise AssertionError("Incorrect source indices in {}".format(csv_path))
            weights = np.array([[float(record["cash_weight"])] +
                                [float(record["risky_weight_{}".format(i+1)]) for i in range(source.shape[1])]
                                for record in table])
            close(np.sum(weights, axis=1), np.ones(len(weights)), "simplex_sum", maximum_errors)
            if np.min(weights) < 0:
                raise AssertionError("Short allocation emitted")
            mu = np.array([float(record["cost_multiplier"]) for record in table])
            turnover = np.array([float(record["risky_dollar_turnover"]) for record in table])
            logs = np.array([float(record["net_log_return"]) for record in table])
            simple = np.array([float(record["net_simple_return"]) for record in table])
            wealth = np.array([float(record["wealth"]) for record in table])
            gross = np.sum(weights*future, axis=1)
            before = np.zeros_like(weights); before[0, 0] = 1.0
            before[1:] = (weights[:-1]*future[:-1])/gross[:-1, None]
            independent_turnover = np.sum(np.abs(mu[:, None]*weights[:, 1:]-before[:, 1:]), axis=1)
            close(turnover, independent_turnover, "turnover", maximum_errors)
            close(mu+row["evaluation_cost"]*independent_turnover, np.ones(len(mu)),
                  "self_financing", maximum_errors)
            close(logs, np.log(mu*gross), "net_log_growth", maximum_errors)
            close(simple, mu*gross-1.0, "net_simple_return", maximum_errors)
            close(wealth, np.cumprod(mu*gross), "wealth_path", maximum_errors)
            close(row["terminal_wealth"], wealth[-1], "terminal_metric", maximum_errors)
            highs = np.maximum.accumulate(np.concatenate(([1.0], wealth)))
            close(row["maximum_drawdown"], np.max(1-np.concatenate(([1.0], wealth))/highs),
                  "drawdown_metric", maximum_errors)
            close(row["mean_risky_dollar_turnover"], np.mean(turnover), "turnover_metric", maximum_errors)
            if run in policy_actions:
                close(weights, policy_actions[run], "cost_sweep_same_actions", maximum_errors)
            else:
                policy_actions[run] = weights
                if row["seed"] is not None:
                    training_runs_checked += 1
                    training = read_json(os.path.join(directory, run+"_training.json"))
                    candidate = max(training["history"], key=lambda h: h["validation_objective"])
                    if candidate["update"] != training["selected_update"]:
                        raise AssertionError("Checkpoint selection differs from validation maximum")
                    checkpoint_zero_selected += int(training["selected_update"] == 0)
                    parameters_path = os.path.join(directory, run+".npz")
                    if training["selected_parameters_sha256"] != digest(parameters_path):
                        raise AssertionError("Selected network parameter hash differs")
                    arrays = np.load(parameters_path)
                    predicted = test_states
                    for layer in range(len(protocol["hidden_sizes"])+1):
                        predicted = np.dot(predicted, arrays["parameter_{}".format(2*layer)]) + \
                            arrays["parameter_{}".format(2*layer+1)]
                        if layer < len(protocol["hidden_sizes"]):
                            predicted = np.maximum(predicted, 0.0)
                        else:
                            predicted = np.exp(predicted-np.max(predicted, axis=1)[:, None])
                            predicted /= np.sum(predicted, axis=1)[:, None]
                    close(predicted, weights, "selected_network_actions", maximum_errors)
            trajectories_checked += 1
    expected_runs = len(protocol["datasets"])*len(protocol["seeds"])*(2+len(protocol["downside_lambdas"]))
    expected_paths = (expected_runs+3*len(protocol["datasets"]))*len(protocol["evaluation_costs"])
    if training_runs_checked != expected_runs or trajectories_checked != expected_paths:
        raise AssertionError("Missing or duplicate expected experiment artifacts")
    return {"status": "PASS", "training_runs_checked": training_runs_checked,
            "trajectories_checked": trajectories_checked,
            "checkpoint_zero_selected": checkpoint_zero_selected,
            "maximum_absolute_errors": maximum_errors,
            "protocol_sha256": digest(protocol_path),
            "scope": "Arithmetic, indices, retained validation selection, parameters and output integrity; not independent market replication"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args()
    report = audit(arguments.protocol, arguments.results)
    with open(arguments.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report, indent=2, sort_keys=True))
