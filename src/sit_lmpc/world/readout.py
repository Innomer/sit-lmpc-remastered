"""Fit the linear read-out from the encoder's whitened z to friction on training cars (condition tag for the safe set).

    python -m sit_lmpc.world.readout --encoder runs/world-encoder-v3
"""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml

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
    np.savez(run / "condition_readout.npz", weights=weights[:-1], bias=weights[-1])
    report = {"r2_test": float(r2), "within_car_std": spread, "train_trajectories": int(len(train_x)),
              "test_trajectories": int(len(test_x))}
    (run / "condition_readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
