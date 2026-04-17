"""
run_fixed_eval.py — fast, reproducible comparison on the held-out test split.
Uses practical evaluation limits so runs complete quickly for submission usage.
"""
from __future__ import annotations
import csv, json, sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "rl_pipeline"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluation_core import (
    evaluate_rl_policy_on_episode_ids,
    evaluate_heuristic_policy_on_episode_ids,
)
from env_factory import load_scheduler_dataset, DEFAULT_RL_DATASET_PATH

MODEL_PATH = REPO_ROOT / "artifacts" / "rl" / "champion_v5_structural_fixes_01" / "final_model.zip"
DATASET_PATH = Path(os.environ.get("RL_DATASET", str(DEFAULT_RL_DATASET_PATH)))
OUTPUT_DIR   = REPO_ROOT / "artifacts" / "rl" / "champion_v5_fixed_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_PATH = REPO_ROOT / "artifacts" / "rl" / "champion_stress_lowwait_ft_20260410" / "episode_splits.json"

SCENARIOS = {
    "stress": {"machine_capacity_scale": 0.35, "machine_pool_size": 32},
    "medium": {"machine_capacity_scale": 0.50, "machine_pool_size": 128},
}
DEADLINE_SLACK    = 2.0
TOP_K             = 16
MAX_STEPS         = 500
MAX_CONSEC_DEFERS = 30
INVALID_LIMIT     = 30
SEED              = 42
HEURISTICS        = ["FCFS", "SJF", "RR"]
UNSCHEDULED_PENALTY = 500.0  # fair-metric penalty for unscheduled tasks


def _aggregate(episodes: list[dict]) -> dict:
    total      = sum(r["total_tasks"]           for r in episodes)
    scheduled  = sum(r["scheduled_tasks"]       for r in episodes)
    unscheduled= sum(r["unscheduled_tasks"]     for r in episodes)
    missed     = sum(r["missed_deadline_tasks"] for r in episodes)
    wait_sum   = sum(r["waiting_time_sum"]      for r in episodes)
    tat_sum    = sum(r["turnaround_time_sum"]   for r in episodes)
    cpu_busy   = sum(r["cpu_busy_time"]         for r in episodes)
    cpu_cap    = sum(r["cpu_capacity_time"]     for r in episodes)
    fair_wait  = wait_sum + UNSCHEDULED_PENALTY * unscheduled
    fair_tat   = tat_sum  + UNSCHEDULED_PENALTY * unscheduled
    return {
        "total_tasks":              total,
        "scheduled_tasks":          scheduled,
        "unscheduled_tasks":        unscheduled,
        "missed_deadline_tasks":    missed,
        "deadline_miss_ratio":      missed / max(total, 1),
        "mean_waiting_time":        wait_sum / max(scheduled, 1),
        "mean_turnaround_time":     tat_sum  / max(scheduled, 1),
        "cpu_utilization":          cpu_busy / max(cpu_cap, 1e-9),
        "fair_mean_waiting_time":   fair_wait / max(total, 1),
        "fair_mean_turnaround_time":fair_tat  / max(total, 1),
    }


def _eval_rl(dataset, episode_ids, scale, pool):
    raw = evaluate_rl_policy_on_episode_ids(
        model_path=MODEL_PATH, dataset=dataset, episode_ids=episode_ids,
        seed=SEED, device="cuda", stochastic=False,
        deadline_slack_factor=DEADLINE_SLACK, top_k_candidates=TOP_K,
        max_steps=MAX_STEPS, max_consecutive_defers=MAX_CONSEC_DEFERS,
        invalid_action_limit=INVALID_LIMIT,
        machine_capacity_scale=scale, machine_pool_size=pool,
        use_hybrid_scheduler=False,
    )
    return _aggregate(raw["episodes"])


def _eval_heuristic(dataset, episode_ids, heuristic, scale, pool):
    raw = evaluate_heuristic_policy_on_episode_ids(
        dataset=dataset, episode_ids=episode_ids, heuristic=heuristic,
        deadline_slack_factor=DEADLINE_SLACK,
        machine_capacity_scale=scale, machine_pool_size=pool,
    )
    return _aggregate(raw["episodes"])


def _print_table(scenario, summary_map):
    print(f"\n{'='*80}")
    print(f"  {scenario.upper()}")
    print(f"{'='*80}")
    fmt = "{:<13} {:>7} {:>10} {:>11} {:>10} {:>11} {:>7} {:>12}"
    print(fmt.format("Scheduler","DMR%","Wait","FairWait","TAT","FairTAT","CPU%","Scheduled"))
    print("-"*85)
    for sched, m in summary_map.items():
        print(fmt.format(
            sched,
            f"{m['deadline_miss_ratio']*100:.2f}%",
            f"{m['mean_waiting_time']:.2f}",
            f"{m['fair_mean_waiting_time']:.2f}",
            f"{m['mean_turnaround_time']:.2f}",
            f"{m['fair_mean_turnaround_time']:.2f}",
            f"{m['cpu_utilization']*100:.2f}%",
            f"{m['scheduled_tasks']}/{m['total_tasks']}",
        ))


def main():
    print("Loading dataset …")
    dataset = load_scheduler_dataset(DATASET_PATH)
    split_payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    episode_ids = sorted(int(episode_id) for episode_id in split_payload["test_episode_ids"])
    print(f"Episodes: {episode_ids}\n")

    all_results = {}
    for scenario, cfg in SCENARIOS.items():
        scale = float(cfg["machine_capacity_scale"])
        pool  = cfg.get("machine_pool_size")
        sm: dict[str, dict] = {}

        for h in HEURISTICS:
            print(f"  [{scenario}] {h} …", flush=True)
            sm[h] = _eval_heuristic(dataset, episode_ids, h, scale, pool)

        print(f"  [{scenario}] RL (max_consec_defers={MAX_CONSEC_DEFERS}) …", flush=True)
        sm["RL Model"] = _eval_rl(dataset, episode_ids, scale, pool)

        _print_table(scenario, sm)
        all_results[scenario] = sm

        # CSV
        fields = ["scheduler","deadline_miss_ratio","mean_waiting_time","fair_mean_waiting_time",
                  "mean_turnaround_time","fair_mean_turnaround_time","cpu_utilization",
                  "scheduled_tasks","unscheduled_tasks","total_tasks"]
        rows = [{f: m.get(f, sched) if f != "scheduler" else sched for f in fields}
                for sched, m in sm.items()]
        csv_path = OUTPUT_DIR / f"comparison_{scenario}_fixed.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"  → {csv_path}")

    # JSON dump
    json_path = OUTPUT_DIR / "all_results_fixed.json"
    json_path.write_text(json.dumps(
        {sc: {k: {kk: float(vv) if isinstance(vv, float) else vv
                  for kk, vv in v.items()}
              for k, v in sm.items()}
         for sc, sm in all_results.items()},
        indent=2))
    print(f"\nJSON: {json_path}")
    print("Done!")


if __name__ == "__main__":
    main()
