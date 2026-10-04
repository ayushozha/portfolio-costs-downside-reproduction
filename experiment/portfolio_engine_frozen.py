"""Historically bounded direct-reinforcement allocation (Python 3.6 / NumPy 1.13).

All-cash initial wealth, proportional fees on risky-dollar trades, cash has a
unit price relative.  This is a feed-forward direct policy, not a reproduction
of Jiang et al.'s CNN, cryptocurrency data, or portfolio-vector memory.
"""
from __future__ import division

import csv
import datetime
import hashlib
import json
import os
import platform
import re
import sys
import time

import numpy as np


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def read_relatives(path):
    with open(path, "r") as handle:
        first_row = next(csv.reader(handle))
    header = bool(first_row) and all(re.match(r"^asset_[0-9]+$", field.strip())
                                    for field in first_row)
    values = np.loadtxt(path, delimiter=",", skiprows=1 if header else 0)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Price relatives must be a positive finite 2-D matrix")
    return values


def make_samples(risky_relatives, lookback):
    """Sample i sees only returns i-lookback:i, then earns return i."""
    n, assets = risky_relatives.shape
    if lookback < 1 or n <= lookback:
        raise ValueError("Insufficient observations for the historical lookback")
    history = np.log(risky_relatives)
    states = np.asarray([history[i-lookback:i].reshape(-1)
                         for i in range(lookback, n)])
    relatives = np.column_stack((np.ones(n-lookback), risky_relatives[lookback:]))
    return states, relatives, np.arange(lookback, n)


def cost_multiplier(weights, before, cost):
    """Solve mu + c sum_risky |mu*w-q| = 1 by monotone bisection."""
    if not 0 <= cost < 1:
        raise ValueError("Proportional fee must lie in [0,1)")
    if cost == 0:
        return np.ones(weights.shape[0])
    # A rebalance can sell and buy, so risky turnover can exceed one.  The
    # triangle bound sum|mu*w-q| <= mu+1 gives a valid lower bracket.
    low = np.full(weights.shape[0], (1.0-cost)/(1.0+cost))
    high = np.ones(weights.shape[0])
    for unused in range(48):
        middle = (low + high) / 2.0
        residual = middle + cost * np.sum(
            np.abs(middle[:, None]*weights[:, 1:] - before[:, 1:]), axis=1) - 1.0
        high = np.where(residual > 0, middle, high)
        low = np.where(residual > 0, low, middle)
    return (low + high) / 2.0


def trajectory(weights, relatives, cost):
    gross = np.sum(weights*relatives, axis=1)
    before = np.zeros_like(weights)
    before[0, 0] = 1.0
    before[1:] = weights[:-1]*relatives[:-1] / gross[:-1, None]
    mu = cost_multiplier(weights, before, cost)
    turnover = np.sum(np.abs(mu[:, None]*weights[:, 1:] - before[:, 1:]), axis=1)
    log_returns = np.log(mu) + np.log(gross)
    return {"gross": gross, "before": before, "mu": mu,
            "turnover": turnover, "log_returns": log_returns,
            "simple_returns": np.expm1(log_returns),
            "wealth": np.exp(np.cumsum(log_returns))}


def allocation_objective(weights, relatives, cost, downside_lambda, reward_start=0):
    """Mean reward and complete current/adjacent-action analytic derivatives.

    For a sampled noninitial block, prepend its predecessor action and set
    reward_start=1.  Its reward is omitted, but next-day trading-cost gradients
    propagate through it.  No preceding-action input is fed to the policy.
    """
    if downside_lambda < 0 or reward_start not in (0, 1):
        raise ValueError("Invalid downside coefficient or predecessor flag")
    tr = trajectory(weights, relatives, cost)
    r = tr["log_returns"]
    negative = np.minimum(r, 0.0)
    rewards = r - downside_lambda*negative**2
    count = len(r)-reward_start
    if count <= 0:
        raise ValueError("The block must contain at least one rewarded sample")
    reward_derivative = 1.0 - 2.0*downside_lambda*negative
    reward_derivative[:reward_start] = 0.0
    reward_derivative /= count
    signs = np.sign(tr["mu"][:, None]*weights[:, 1:] - tr["before"][:, 1:])
    denominator = 1.0 + cost*np.sum(signs*weights[:, 1:], axis=1)
    gradient = relatives/tr["gross"][:, None]
    gradient[:, 1:] -= cost*signs/denominator[:, None]
    gradient *= reward_derivative[:, None]
    before_gradient = np.zeros_like(weights)
    before_gradient[:, 1:] = cost*signs/(tr["mu"]*denominator)[:, None]
    before_gradient *= reward_derivative[:, None]
    # q[t] = w[t-1]*y[t-1] / <w[t-1],y[t-1]>.
    centered = before_gradient[1:] - np.sum(
        before_gradient[1:]*tr["before"][1:], axis=1)[:, None]
    gradient[:-1] += relatives[:-1]/tr["gross"][:-1, None]*centered
    return float(np.mean(rewards[reward_start:])), gradient, tr


class Policy(object):
    def __init__(self, input_size, assets_with_cash, hidden, seed):
        rng = np.random.RandomState(seed)
        sizes = [input_size] + list(hidden) + [assets_with_cash]
        self.parameters = []
        for index in range(len(sizes)-1):
            scale = np.sqrt(2.0/sizes[index]) if index < len(sizes)-2 else 0.01
            self.parameters.extend([rng.randn(sizes[index], sizes[index+1])*scale,
                                    np.zeros(sizes[index+1])])

    def forward(self, states):
        activations = [states]
        preactivations = []
        for layer in range(len(self.parameters)//2):
            z = np.dot(activations[-1], self.parameters[2*layer]) + self.parameters[2*layer+1]
            preactivations.append(z)
            if layer == len(self.parameters)//2-1:
                shifted = z-np.max(z, axis=1)[:, None]
                exponential = np.exp(shifted)
                activations.append(exponential/np.sum(exponential, axis=1)[:, None])
            else:
                activations.append(np.maximum(z, 0.0))
        return activations[-1], (activations, preactivations)

    def backward(self, allocation_gradient, cache):
        activations, preactivations = cache
        weights = activations[-1]
        delta = weights*(allocation_gradient - np.sum(
            allocation_gradient*weights, axis=1)[:, None])
        gradients = [None]*len(self.parameters)
        for layer in reversed(range(len(self.parameters)//2)):
            gradients[2*layer] = np.dot(activations[layer].T, delta)
            gradients[2*layer+1] = np.sum(delta, axis=0)
            if layer:
                delta = np.dot(delta, self.parameters[2*layer].T)
                delta *= preactivations[layer-1] > 0
        return gradients

    def copy_parameters(self):
        return [parameter.copy() for parameter in self.parameters]


class AdamV1(object):
    """Kingma/Ba arXiv:1412.6980v1 Algorithm 1, innovation-fraction notation."""
    def __init__(self, parameters, learning_rate=0.0002):
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.epsilon = 0.1, 0.001, 1e-8
        self.first = [np.zeros_like(p) for p in parameters]
        self.second = [np.zeros_like(p) for p in parameters]
        self.step = 0

    def ascend(self, parameters, gradients):
        self.step += 1
        for index, gradient in enumerate(gradients):
            self.first[index] = self.beta1*gradient + (1-self.beta1)*self.first[index]
            self.second[index] = self.beta2*gradient**2 + (1-self.beta2)*self.second[index]
            first = self.first[index]/(1-(1-self.beta1)**self.step)
            second = self.second[index]/(1-(1-self.beta2)**self.step)
            parameters[index] += self.learning_rate*first/(np.sqrt(second)+self.epsilon)


def train_policy(train_x, train_y, validation_x, validation_y, config, seed):
    policy = Policy(train_x.shape[1], train_y.shape[1], config["hidden_sizes"], seed)
    optimizer = AdamV1(policy.parameters, config["learning_rate"])
    rng = np.random.RandomState(seed+10000)
    history = []
    best = None
    best_score = -np.inf
    selected_update = None
    batch = min(config["batch_size"], len(train_x))
    for update in range(config["updates"]+1):
        if update % config["checkpoint_interval"] == 0 or update == config["updates"]:
            w, unused = policy.forward(validation_x)
            score, unused_gradient, unused_trajectory = allocation_objective(
                w, validation_y, config["train_cost"], config["downside_lambda"])
            train_w, unused = policy.forward(train_x)
            train_score, unused_gradient, unused_trajectory = allocation_objective(
                train_w, train_y, config["train_cost"], config["downside_lambda"])
            history.append({"update": update, "train_objective": train_score,
                            "validation_objective": score})
            if score > best_score:
                best_score, selected_update = score, update
                best = policy.copy_parameters()
        if update == config["updates"]:
            break
        start = rng.randint(0, len(train_x)-batch+1)
        predecessor = 1 if start else 0
        lo, hi = start-predecessor, start+batch
        allocations, cache = policy.forward(train_x[lo:hi])
        unused_score, gradient, unused_trajectory = allocation_objective(
            allocations, train_y[lo:hi], config["train_cost"],
            config["downside_lambda"], predecessor)
        optimizer.ascend(policy.parameters, policy.backward(gradient, cache))
    policy.parameters = best
    return policy, history, selected_update, best_score


def metrics(tr, annualization=252):
    simple, wealth = tr["simple_returns"], tr["wealth"]
    mean = float(np.mean(simple))
    deviation = float(np.std(simple, ddof=1)) if len(simple)>1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(simple, 0.0)**2)))
    running_high = np.maximum.accumulate(np.concatenate(([1.0], wealth)))
    drawdown = 1.0 - np.concatenate(([1.0], wealth))/running_high
    return {"observations": int(len(simple)), "terminal_wealth": float(wealth[-1]),
            "cumulative_return": float(wealth[-1]-1.0),
            "mean_daily_log_return": float(np.mean(tr["log_returns"])),
            "annualized_return": float(np.exp(np.mean(tr["log_returns"])*annualization)-1.0),
            "annualized_volatility": deviation*np.sqrt(annualization),
            "annualized_downside_deviation": downside*np.sqrt(annualization),
            "annualized_sharpe_zero_cash_rate": mean/deviation*np.sqrt(annualization) if deviation else None,
            "annualized_sortino_zero_target": mean/downside*np.sqrt(annualization) if downside else None,
            "maximum_drawdown": float(np.max(drawdown)),
            "mean_risky_dollar_turnover": float(np.mean(tr["turnover"])),
            "total_log_fee_drag": float(-np.sum(np.log(tr["mu"]))),
            "negative_return_fraction": float(np.mean(simple<0))}


def baseline_weights(relatives, name):
    n, d = relatives.shape
    weights = np.zeros((n, d))
    if name == "cash":
        weights[:, 0] = 1.0
    elif name == "equal_weight_crp":
        weights[:, 1:] = 1.0/(d-1)
    elif name == "equal_weight_buy_hold":
        weights[0, 1:] = 1.0/(d-1)
        for index in range(1, n):
            holdings = weights[index-1]*relatives[index-1]
            weights[index] = holdings/np.sum(holdings)
    else:
        raise ValueError("Unknown baseline")
    return weights


def write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def write_trajectory(path, weights, tr, source_indices):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_return_index_zero_based", "net_log_return", "net_simple_return",
                         "wealth", "cost_multiplier", "risky_dollar_turnover", "cash_weight"]+
                        ["risky_weight_{}".format(i+1) for i in range(weights.shape[1]-1)])
        for index in range(len(weights)):
            writer.writerow([int(source_indices[index]), float(tr["log_returns"][index]),
                             float(tr["simple_returns"][index]), float(tr["wealth"][index]),
                             float(tr["mu"][index]), float(tr["turnover"][index])]+weights[index].tolist())


def run_protocol(protocol_path, output_directory):
    started = datetime.datetime.utcnow().isoformat()+"Z"
    with open(protocol_path) as handle:
        protocol = json.load(handle)
    expected_runtime = protocol.get("scientific_runtime", {})
    if expected_runtime.get("python_version") != platform.python_version() or \
            expected_runtime.get("numpy_version") != np.__version__:
        raise ValueError("The scientific runtime does not match the frozen historical releases")
    if os.path.exists(output_directory):
        raise ValueError("Refusing to replace an existing experiment output directory")
    os.makedirs(output_directory)
    all_metrics = []
    for dataset in protocol["datasets"]:
        risky = read_relatives(dataset["path"])
        states, relatives, source_indices = make_samples(risky, protocol["lookback"])
        # Splits are ORIGINAL return-row indices, not post-lookback indices.
        train_mask = (source_indices>=dataset["train"][0]) & (source_indices<dataset["train"][1])
        validation_mask = (source_indices>=dataset["validation"][0]) & (source_indices<dataset["validation"][1])
        test_mask = (source_indices>=dataset["test"][0]) & (source_indices<dataset["test"][1])
        if not (np.any(train_mask) and np.any(validation_mask) and np.any(test_mask)):
            raise ValueError("Every chronological partition must contain samples")
        if not (dataset["train"][1] <= dataset["validation"][0] and
                dataset["validation"][1] <= dataset["test"][0] and
                dataset["test"][1] <= len(risky)):
            raise ValueError("Chronological splits overlap or exceed observations")
        center = np.mean(states[train_mask], axis=0)
        scale = np.maximum(np.std(states[train_mask], axis=0), 1e-6)
        states = (states-center)/scale
        dataset_dir = os.path.join(output_directory, dataset["id"])
        os.makedirs(dataset_dir)
        np.savez(os.path.join(dataset_dir, "training_standardization.npz"), center=center, scale=scale)
        source_record = {"shape": list(risky.shape), "data_sha256": sha256(dataset["path"]),
                         "dataset": dataset, "partition_sample_counts": {
                             "train": int(np.sum(train_mask)), "validation": int(np.sum(validation_mask)),
                             "test": int(np.sum(test_mask))}, "actual_start_utc": started}
        write_json(os.path.join(dataset_dir, "data_record.json"), source_record)
        test_y, test_indices = relatives[test_mask], source_indices[test_mask]
        for name in ["cash", "equal_weight_crp", "equal_weight_buy_hold"]:
            weights = baseline_weights(test_y, name)
            for cost in protocol["evaluation_costs"]:
                tr = trajectory(weights, test_y, cost)
                row = metrics(tr, protocol["annualization"])
                row.update({"dataset": dataset["id"], "policy": name, "seed": None,
                            "evaluation_cost": cost, "downside_lambda": None})
                all_metrics.append(row)
                write_trajectory(os.path.join(dataset_dir, "{}_cost{}.csv".format(name, cost)),
                                 weights, tr, test_indices)
        variants = [{"name": "return_only", "train_cost": 0.0, "downside_lambda": 0.0},
                    {"name": "cost_aware", "train_cost": protocol["train_cost"], "downside_lambda": 0.0}]
        variants += [{"name": "downside_lambda{}".format(lam), "train_cost": protocol["train_cost"],
                      "downside_lambda": lam} for lam in protocol["downside_lambdas"]]
        for variant in variants:
            config = dict(protocol)
            config.update(variant)
            for seed in protocol["seeds"]:
                run_started = time.time()
                run_started_utc = datetime.datetime.utcnow().isoformat()+"Z"
                policy, history, update, score = train_policy(
                    states[train_mask], relatives[train_mask], states[validation_mask],
                    relatives[validation_mask], config, seed)
                run_name = "{}_seed{}".format(variant["name"], seed)
                arrays = dict(("parameter_{}".format(i), p) for i, p in enumerate(policy.parameters))
                np.savez(os.path.join(dataset_dir, run_name+".npz"), **arrays)
                write_json(os.path.join(dataset_dir, run_name+"_training.json"), {
                    "seed": seed, "config": variant, "history": history,
                    "selected_update": update, "selected_validation_objective": score,
                    "elapsed_seconds": time.time()-run_started,
                    "actual_start_utc": run_started_utc,
                    "selected_parameters_sha256": sha256(os.path.join(dataset_dir, run_name+".npz")),
                    "actual_completed_utc": datetime.datetime.utcnow().isoformat()+"Z"})
                weights, unused = policy.forward(states[test_mask])
                for cost in protocol["evaluation_costs"]:
                    tr = trajectory(weights, test_y, cost)
                    row = metrics(tr, protocol["annualization"])
                    row.update({"dataset": dataset["id"], "policy": variant["name"], "seed": seed,
                                "evaluation_cost": cost, "downside_lambda": variant["downside_lambda"],
                                "selected_update": update})
                    all_metrics.append(row)
                    write_trajectory(os.path.join(dataset_dir, "{}_cost{}.csv".format(run_name, cost)),
                                     weights, tr, test_indices)
                print("{} {} selected_update={} validation={:.8f}".format(
                    dataset["id"], run_name, update, score), flush=True)
    write_json(os.path.join(output_directory, "metrics.json"), all_metrics)
    write_json(os.path.join(output_directory, "execution.json"), {
        "actual_start_utc": started, "actual_completed_utc": datetime.datetime.utcnow().isoformat()+"Z",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scientific_runtime": expected_runtime,
        "scientific_runtime_source_ids": protocol.get("scientific_runtime_source_ids", []),
        "compute_device": "CPU", "cuda_used": False,
        "platform": platform.platform(), "protocol_path": os.path.abspath(protocol_path),
        "protocol_sha256": sha256(protocol_path), "engine_sha256": sha256(__file__),
        "scientific_result_rows": len(all_metrics), "fictional_historical_execution": False,
        "arguments": sys.argv})
    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run_protocol(arguments.protocol, arguments.output)
