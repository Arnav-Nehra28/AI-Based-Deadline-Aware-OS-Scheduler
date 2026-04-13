from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .env_factory import (
        DEFAULT_RL_DATASET_PATH,
        build_scheduler_env,
        load_scheduler_dataset,
        subset_dataset_by_episode_ids,
    )
    from .wrappers import ActionMaskInfoWrapper
except ImportError:
    from env_factory import (
        DEFAULT_RL_DATASET_PATH,
        build_scheduler_env,
        load_scheduler_dataset,
        subset_dataset_by_episode_ids,
    )
    from wrappers import ActionMaskInfoWrapper


def _require_rl_dependencies() -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing RL dependency for policy evaluation. Install with:\n"
            "pip install gymnasium stable-baselines3 sb3-contrib"
        ) from exc
    return MaskablePPO


def _parse_episode_ids(raw_value: str | None) -> list[int]:
    if raw_value is None or not raw_value.strip():
        return []

    episode_ids: list[int] = []
    for chunk in raw_value.split(","):
        value = chunk.strip()
        if not value:
            continue
        episode_ids.append(int(value))
    return sorted(set(episode_ids))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO scheduler policy.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RL_DATASET_PATH)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-ids", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/rl/eval/latest_eval.json"))
    parser.add_argument("--deadline-slack-factor", type=float, default=2.0)

    parser.add_argument("--top-k-candidates", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-consecutive-defers", type=int, default=30)
    parser.add_argument("--invalid-action-limit", type=int, default=30)
    parser.add_argument("--machine-capacity-scale", type=float, default=1.0)
    parser.add_argument("--machine-pool-size", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    MaskablePPO = _require_rl_dependencies()

    dataset = load_scheduler_dataset(args.dataset)
    available_episode_ids = sorted(dataset.episodes["episode_id"].astype(int).unique().tolist())
    if not available_episode_ids:
        raise ValueError("Dataset does not contain any episodes for evaluation.")

    requested_episode_ids = _parse_episode_ids(args.episode_ids)
    if requested_episode_ids:
        missing_ids = sorted(set(requested_episode_ids) - set(available_episode_ids))
        if missing_ids:
            raise ValueError(f"Requested episode ids are missing in the dataset: {missing_ids}")
        selected_episode_ids = requested_episode_ids
    else:
        selected_count = max(1, int(args.episodes))
        selected_episode_ids = available_episode_ids[:selected_count]

    eval_dataset = subset_dataset_by_episode_ids(dataset, selected_episode_ids)
    env = ActionMaskInfoWrapper(
        build_scheduler_env(
            dataset=eval_dataset,
            top_k_candidates=int(args.top_k_candidates),
            max_steps=int(args.max_steps),
            max_consecutive_defers=int(args.max_consecutive_defers),
            invalid_action_limit=int(args.invalid_action_limit),
            machine_capacity_scale=float(args.machine_capacity_scale),
            machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
            deadline_slack_factor=float(args.deadline_slack_factor),
            random_state=int(args.seed),
            randomize_on_reset=False,
        )
    )
    model = MaskablePPO.load(str(args.model_path), device=str(args.device))

    episode_metrics: list[dict[str, Any]] = []
    deterministic = not bool(args.stochastic)

    for episode_idx, episode_id in enumerate(selected_episode_ids):
        observation, _ = env.reset(
            seed=int(args.seed) + episode_idx,
            options={"episode_id": int(episode_id)},
        )
        done = False
        episode_reward = 0.0
        episode_steps = 0
        feasible_count = 0
        defer_count = 0
        invalid_count = 0
        terminated = False
        truncated = False

        while not done:
            action_mask = env.action_masks()
            action, _ = model.predict(
                observation,
                deterministic=deterministic,
                action_masks=action_mask,
            )
            action_value = int(np.asarray(action).item())

            observation, reward, terminated, truncated, info = env.step(action_value)
            reward_components = info.get("reward_components")
            if not isinstance(reward_components, dict):
                reward_components = {}

            episode_reward += float(reward)
            episode_steps += 1

            if bool(info.get("was_feasible", False)):
                feasible_count += 1
            elif "defer_penalty" in reward_components:
                defer_count += 1
            else:
                invalid_count += 1

            done = bool(terminated or truncated)

        step_denominator = max(episode_steps, 1)
        episode_metrics.append(
            {
                "episode_id": int(episode_id),
                "total_reward": float(episode_reward),
                "steps": int(episode_steps),
                "feasible_count": int(feasible_count),
                "defer_count": int(defer_count),
                "invalid_count": int(invalid_count),
                "feasible_rate": float(feasible_count / step_denominator),
                "defer_rate": float(defer_count / step_denominator),
                "invalid_rate": float(invalid_count / step_denominator),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    mean_reward = float(np.mean([episode["total_reward"] for episode in episode_metrics]))
    mean_feasible_rate = float(np.mean([episode["feasible_rate"] for episode in episode_metrics]))
    mean_defer_rate = float(np.mean([episode["defer_rate"] for episode in episode_metrics]))
    mean_invalid_rate = float(np.mean([episode["invalid_rate"] for episode in episode_metrics]))

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset),
        "deterministic": deterministic,
        "episode_count": len(episode_metrics),
        "selected_episode_ids": selected_episode_ids,
        "summary": {
            "mean_total_reward": mean_reward,
            "mean_feasible_rate": mean_feasible_rate,
            "mean_defer_rate": mean_defer_rate,
            "mean_invalid_rate": mean_invalid_rate,
            "terminated_episodes": int(sum(1 for episode in episode_metrics if episode["terminated"])),
            "truncated_episodes": int(sum(1 for episode in episode_metrics if episode["truncated"])),
        },
        "episodes": episode_metrics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    env.close()

    print(f"Evaluation complete over {len(episode_metrics)} episode(s).")
    print(f"Mean total reward: {mean_reward:.4f}")
    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
