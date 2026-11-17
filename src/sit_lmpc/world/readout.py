"""Condition tags for the safe set, saved next to an encoder run:
- a linear read-out from the encoder's whitened z to friction, fitted on training cars (needs sim labels);
- a functional distance: how differently two z make the transition predict on fixed probe states (no labels).

    python -m sit_lmpc.world.readout --encoder runs/world-encoder-v3
"""

import argparse
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import yaml

from dyna_dist.inference.posterior import components
from dyna_dist.io.checkpoint import load_model
from dyna_dist.models.features import prior_moments_and_variance

from sit_lmpc.world.data import WorldData
from sit_lmpc.world.runtime import EncoderBelief


def trajectory_features(belief, data, burn=10, minimum=60):
    instances = {member["instance_id"]: member for member in data.dataset.instances}
    features, friction, sources = [], [], []
    for item in data.items:
        if item["kind"] != "instance" or len(item["controls"]) < minimum:
            continue
        belief.reset()
        whitened = [belief.step(observation)[3] for observation in item["observations"]]
        features.append(np.mean(whitened[burn:], axis=0))
        friction.append(instances[item["source"]]["friction"])
        sources.append(item["source"])
    return np.asarray(features), np.asarray(friction), np.asarray(sources)


FUNCTIONAL_COMPONENTS = (0, 1, 3, 4, 5, 6)


def functional_matrix(transition, data, count=256, seed=0):
    """P such that ||P (w1 - w2)|| / sqrt(rows) is the RMS difference of normalized predicted increments on probes."""
    rng = np.random.default_rng(seed)
    states, controls = [], []
    for item in data.items:
        moving = np.flatnonzero(item["states"][:-1, 3] > 1.0)
        if len(moving):
            chosen = rng.choice(moving, min(4, len(moving)), replace=False)
            states.append(item["states"][chosen])
            controls.append(item["controls"][chosen])
    states, controls = np.concatenate(states), np.concatenate(controls)
    pick = rng.choice(len(states), min(count, len(states)), replace=False)
    _, basis = components(transition, jnp.asarray(states[pick]), jnp.asarray(controls[pick]))
    rows = np.asarray(basis)[:, list(FUNCTIONAL_COMPONENTS), :].reshape(-1, basis.shape[-1])
    _, factor, _ = prior_moments_and_variance(transition.parameters, transition.config)
    return rows @ np.asarray(factor), states[pick], controls[pick]


def functional_distance(matrix, w1, w2):
    return float(np.linalg.norm(matrix @ (np.asarray(w1) - np.asarray(w2))) / np.sqrt(len(matrix)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()
    run = Path(args.encoder)
    config = yaml.safe_load((run / "config.yaml").read_text())
    belief = EncoderBelief(run)
    train_x, train_y, _ = trajectory_features(belief, WorldData(config["dataset"], "train", include_changes=False))
    test_x, test_y, test_sources = trajectory_features(belief, WorldData(config["dataset"], "test", include_changes=False))
    weights, *_ = np.linalg.lstsq(np.column_stack((train_x, np.ones(len(train_x)))), train_y, rcond=None)
    predicted = np.column_stack((test_x, np.ones(len(test_x)))) @ weights
    r2 = 1 - np.sum((test_y - predicted) ** 2) / np.sum((test_y - test_y.mean()) ** 2)
    spread = {str(car): float(predicted[test_sources == car].std()) for car in np.unique(test_sources)}
    train = WorldData(config["dataset"], "train", include_changes=False)
    matrix, probe_states, probe_controls = functional_matrix(load_model(config["transition"]), train)
    np.savez(run / "condition_readout.npz", weights=weights[:-1], bias=weights[-1], functional=matrix,
             probe_states=probe_states, probe_controls=probe_controls)
    report = {"r2_test": float(r2), "within_car_std": spread, "train_trajectories": int(len(train_x)),
              "test_trajectories": int(len(test_x))}
    (run / "condition_readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
