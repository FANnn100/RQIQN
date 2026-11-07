"""Aggregate navigation evaluation episodes with equal weight per training seed."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


METRICS = (
    "success_rate", "collision_rate", "timeout_rate", "out_of_boundary_rate",
    "discounted_return", "raw_return", "time_success", "energy_success",
)

TABLE1_COLUMNS = (
    ("Success (%)", "success_rate", 100.0),
    ("Collision (%)", "collision_rate", 100.0),
    ("Timeout (%)", "timeout_rate", 100.0),
    ("Return (discounted)", "discounted_return", 1.0),
    ("Time_succ (s)", "time_success", 1.0),
    ("Energy_succ", "energy_success", 1.0),
)


def aggregate_episodes(episodes):
    """Compute one seed's metrics; time/energy condition on its own successes."""
    if not episodes:
        raise ValueError("Cannot aggregate an empty episode list")
    successes = [episode for episode in episodes if episode["success"]]
    result = {
        "episodes": len(episodes),
        "successful_episodes": len(successes),
        "success_rate": statistics.mean(float(e["success"]) for e in episodes),
        "collision_rate": statistics.mean(float(e["collision"]) for e in episodes),
        "timeout_rate": statistics.mean(float(e["timeout"]) for e in episodes),
        "out_of_boundary_rate": statistics.mean(
            float(e.get("out_of_boundary", False)) for e in episodes),
        "discounted_return": statistics.mean(e["discounted_return"] for e in episodes),
        "raw_return": statistics.mean(e["raw_return"] for e in episodes),
        "time_success": statistics.mean(e["time"] for e in successes) if successes else None,
        "energy_success": statistics.mean(e["energy"] for e in successes) if successes else None,
    }
    return result


def summarize_seeds(per_seed_metrics):
    """Use unweighted seed means and sample SD (ddof=1), never pooled episodes.

    A seed without successes has undefined conditional time and energy. Exclude
    those values only for those metrics and report the effective seed count.
    """
    if not per_seed_metrics:
        raise ValueError("At least one seed is required")
    summary = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in per_seed_metrics.values()
                  if row[metric] is not None and math.isfinite(float(row[metric]))]
        summary[metric] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "n_seeds": len(values),
        }
    return summary


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def aggregate_runs(run_dirs):
    """Read fresh evaluation output and reject duplicate seeds or mixed layouts."""
    if not run_dirs:
        raise ValueError("At least one evaluation directory is required")
    per_policy = {}
    reference = None
    sources = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        settings = json.loads((run_dir / "settings.json").read_text())
        signature = tuple(settings.get(key) for key in
                          ("protocol", "layout_sha256", "num_episodes", "max_steps", "smoke"))
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Runs use different layouts, horizons, or smoke settings")
        seed = str(settings["training_seed"])
        episodes_by_policy = {}
        episode_ids = {}
        with (run_dir / "episodes.jsonl").open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                policy = row["policy"]
                ids = episode_ids.setdefault(policy, set())
                if row["episode_index"] in ids:
                    raise ValueError("Duplicate episode index in " + str(run_dir))
                ids.add(row["episode_index"])
                episodes_by_policy.setdefault(policy, []).append(row)
        if set(episodes_by_policy) != set(settings["policies"]):
            raise ValueError("Incomplete evaluation policies in " + str(run_dir))
        for policy, episodes in episodes_by_policy.items():
            if len(episodes) != settings["num_episodes"]:
                raise ValueError("Incomplete evaluation episodes in " + str(run_dir))
            seeds = per_policy.setdefault(policy, {})
            if seed in seeds:
                raise ValueError("Duplicate training seed for policy " + policy + ": " + seed)
            seeds[seed] = aggregate_episodes(episodes)
        sources.append(str(run_dir))
    return {
        "schema_version": 1,
        "source_runs": sources,
        "protocol": reference[0],
        "layout_sha256": reference[1],
        "num_episodes": reference[2],
        "max_steps": reference[3],
        "smoke": reference[4],
        "weighting": "equal weight per training seed; sample standard deviation (ddof=1)",
        "conditional_metrics": "time and energy averaged within each seed's successful episodes first",
        "per_seed_metrics": per_policy,
        "summary": {policy: summarize_seeds(seeds) for policy, seeds in per_policy.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = aggregate_runs(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", result)
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "training_seed", "episodes", "successful_episodes"] + list(METRICS))
        for policy, seeds in result["per_seed_metrics"].items():
            for seed, metrics in sorted(seeds.items()):
                writer.writerow([policy, seed, metrics["episodes"], metrics["successful_episodes"]]
                                + [metrics[key] for key in METRICS])
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "metric", "mean", "std", "n_seeds"])
        for policy, metrics in result["summary"].items():
            for key, value in metrics.items():
                writer.writerow([policy, key, value["mean"], value["std"], value["n_seeds"]])
    with (args.output_dir / "table1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Policy"] + [label for label, _, _ in TABLE1_COLUMNS])
        for policy in ("full", "adaptive"):
            if policy not in result["summary"]:
                continue
            cells = []
            for _, metric, multiplier in TABLE1_COLUMNS:
                values = result["summary"][policy][metric]
                if values["mean"] is None or values["std"] is None:
                    cells.append("NA")
                else:
                    cells.append("{:.2f} ± {:.2f}".format(
                        multiplier * values["mean"], multiplier * values["std"]))
            writer.writerow([policy] + cells)
    print("Saved seed-weighted summary:", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
