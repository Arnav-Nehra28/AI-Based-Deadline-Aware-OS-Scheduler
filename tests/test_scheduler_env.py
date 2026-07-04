from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rl_pipeline.env_dataset import RLEnvDataset, load_env_dataset, save_env_dataset
from rl_pipeline.environment import TaskSchedulingEnv
from rl_pipeline.gym_compat import GYMNASIUM_AVAILABLE
from rl_pipeline.validate_environment import validate_environment_behavior


def build_synthetic_dataset() -> RLEnvDataset:
    machines = pd.DataFrame(
        [
            {"machine_id": "m1", "cpu_capacity": 1.0, "mem_capacity": 1.0, "disk_capacity": 1.0},
            {"machine_id": "m2", "cpu_capacity": 0.8, "mem_capacity": 0.8, "disk_capacity": 0.8},
        ]
    )
    tasks = pd.DataFrame(
        [
            {
                "episode_id": 0,
                "task_index": 0,
                "task_id": "t0",
                "arrival_time": 0.0,
                "duration": 4.0,
                "cpu_demand": 0.4,
                "mem_demand": 0.3,
                "disk_demand": 0.2,
                "historical_machine_id": "m1",
            },
            {
                "episode_id": 0,
                "task_index": 1,
                "task_id": "t1",
                "arrival_time": 1.0,
                "duration": 2.0,
                "cpu_demand": 0.3,
                "mem_demand": 0.2,
                "disk_demand": 0.2,
                "historical_machine_id": "m2",
            },
            {
                "episode_id": 0,
                "task_index": 2,
                "task_id": "t2",
                "arrival_time": 5.0,
                "duration": 1.0,
                "cpu_demand": 0.2,
                "mem_demand": 0.2,
                "disk_demand": 0.1,
                "historical_machine_id": "m1",
            },
            {
                "episode_id": 1,
                "task_index": 0,
                "task_id": "t3",
                "arrival_time": 0.0,
                "duration": 1.0,
                "cpu_demand": 0.2,
                "mem_demand": 0.2,
                "disk_demand": 0.1,
                "historical_machine_id": "m1",
            },
        ]
    )
    episodes = pd.DataFrame(
        [
            {"episode_id": 0, "task_count": 3, "start_time": 0.0, "end_time": 6.0, "source_kind": "synthetic"},
            {"episode_id": 1, "task_count": 1, "start_time": 0.0, "end_time": 1.0, "source_kind": "synthetic"},
        ]
    )
    return RLEnvDataset(
        tasks=tasks,
        machines=machines,
        episodes=episodes,
        metadata={"source_kind": "synthetic", "episode_length": 3},
    )


class TaskSchedulingEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = build_synthetic_dataset()

    def _make_env(self, **kwargs: object) -> TaskSchedulingEnv:
        env_kwargs = {
            "dataset": self.dataset,
            "top_k_candidates": 4,
            "max_steps": 10,
            "max_consecutive_defers": 4,
            "invalid_action_limit": 4,
            "randomize_on_reset": False,
        }
        env_kwargs.update(kwargs)
        return TaskSchedulingEnv(**env_kwargs)

    def _pick_first_feasible_action(self, observation: dict[str, np.ndarray]) -> int:
        feasible_mask = observation["candidate_features"][:, 1] > 0.5
        feasible_indices = np.where(feasible_mask)[0]
        self.assertGreater(len(feasible_indices), 0, "Expected at least one feasible machine candidate.")
        return int(feasible_indices[0])

    def test_reset_is_seed_deterministic(self) -> None:
        env_a = self._make_env(randomize_on_reset=True)
        env_b = self._make_env(randomize_on_reset=True)

        obs_a, info_a = env_a.reset(seed=7)
        obs_b, info_b = env_b.reset(seed=7)

        self.assertEqual(info_a["episode_id"], info_b["episode_id"])
        np.testing.assert_allclose(obs_a["task_features"], obs_b["task_features"])
        np.testing.assert_allclose(obs_a["candidate_features"], obs_b["candidate_features"])
        np.testing.assert_allclose(obs_a["fleet_summary"], obs_b["fleet_summary"])

    def test_step_follows_gymnasium_contract(self) -> None:
        env = self._make_env()
        observation, info = env.reset(options={"episode_id": 0})
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIn("action_mask", info)
        self.assertIn("candidate_machine_ids", info)

        action = self._pick_first_feasible_action(observation)
        result = env.step(action)
        self.assertEqual(len(result), 5)
        next_observation, reward, terminated, truncated, next_info = result
        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("reward_components", next_info)
        self.assertIn("current_time", next_info)

    def test_action_mask_shape_and_non_empty(self) -> None:
        env = self._make_env()
        _, info = env.reset(options={"episode_id": 0})
        self.assertEqual(info["action_mask"].shape, (5,))
        self.assertGreater(int(info["action_mask"].sum()), 0)

    def test_feasible_assignment_consumes_machine_capacity(self) -> None:
        env = self._make_env()
        observation, _ = env.reset(options={"episode_id": 0})
        action = self._pick_first_feasible_action(observation)
        machine_index = env._candidate_machine_indices[action]
        before = env.machine_residual[machine_index].copy()

        env.step(action)

        after = env.machine_residual[machine_index].copy()
        self.assertTrue(np.any(after < before))

    def test_completed_jobs_release_capacity_when_time_advances(self) -> None:
        env = self._make_env(max_consecutive_defers=10, max_steps=20)
        observation, _ = env.reset(options={"episode_id": 0})
        action = self._pick_first_feasible_action(observation)
        machine_index = env._candidate_machine_indices[action]
        env.step(action)

        depleted = env.machine_residual[machine_index].copy()
        self.assertTrue(np.any(depleted < env.machine_capacities[machine_index]))

        while env.current_time < 4.0:
            env.step(env.defer_action)

        np.testing.assert_allclose(env.machine_residual[machine_index], env.machine_capacities[machine_index])

    def test_defer_advances_time_and_applies_negative_reward(self) -> None:
        env = self._make_env()
        _, info = env.reset(options={"episode_id": 0})
        before_time = info["current_time"]

        _, reward, terminated, truncated, next_info = env.step(env.defer_action)

        self.assertLess(reward, 0.0)
        self.assertEqual(next_info["current_time"], 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        second_before_time = next_info["current_time"]
        _, second_reward, _, _, second_info = env.step(env.defer_action)
        self.assertLess(second_reward, 0.0)
        self.assertEqual(second_info["current_time"], second_before_time)

    def test_time_moves_only_to_real_external_events(self) -> None:
        env = self._make_env(max_consecutive_defers=10, max_steps=20)
        _, info = env.reset(options={"episode_id": 0})
        self.assertEqual(info["current_time"], 0.0)

        observation_after_defer, _, _, _, after_first_defer = env.step(env.defer_action)
        self.assertEqual(after_first_defer["current_time"], 1.0)

        action = self._pick_first_feasible_action(observation_after_defer)
        _, _, _, _, after_assignment = env.step(action)
        self.assertIn(after_assignment["current_time"], {1.0, 5.0})

    def test_shortlist_stays_fixed_shape_when_machine_pool_is_smaller_than_top_k(self) -> None:
        env = self._make_env()
        observation, _ = env.reset(options={"episode_id": 0})
        self.assertEqual(observation["candidate_features"].shape, (4, env.CANDIDATE_FEATURE_DIM))
        self.assertEqual(len(env._candidate_machine_ids), 4)

    def test_observation_dimensions_include_deadline_and_completion_features(self) -> None:
        env = self._make_env()
        observation, _ = env.reset(options={"episode_id": 0})
        self.assertEqual(observation["task_features"].shape, (env.TASK_FEATURE_DIM,))
        self.assertEqual(observation["fleet_summary"].shape, (env.FLEET_SUMMARY_DIM,))
        self.assertGreaterEqual(float(observation["task_features"][8]), 0.0)
        self.assertGreaterEqual(float(observation["task_features"][9]), 0.0)

    def test_effective_max_steps_scales_with_episode_size(self) -> None:
        env = self._make_env(max_steps=5)
        env.reset(options={"episode_id": 0})
        self.assertEqual(env._effective_max_steps, 18)

    def test_feasible_assignment_adds_deadline_bonus_when_on_time(self) -> None:
        env = self._make_env()
        observation, _ = env.reset(options={"episode_id": 1})
        action = self._pick_first_feasible_action(observation)
        _, _, _, _, info = env.step(action)
        self.assertIn("deadline_met_bonus", info["reward_components"])

    def test_terminal_completion_bonus_tracks_scheduled_tasks(self) -> None:
        trunc_env = self._make_env(max_steps=20, max_consecutive_defers=1, invalid_action_limit=1)
        trunc_env.reset(options={"episode_id": 0})
        _, _, _, truncated, trunc_info = trunc_env.step(999)
        self.assertTrue(truncated)
        self.assertEqual(trunc_info["reward_components"]["terminal_completion_bonus"], -2.0)
        self.assertEqual(trunc_info["reward_components"]["terminal_on_time_bonus"], -5.0)

        term_env = self._make_env()
        observation, _ = term_env.reset(options={"episode_id": 1})
        action = self._pick_first_feasible_action(observation)
        _, _, terminated, _, term_info = term_env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(term_info["reward_components"]["terminal_completion_bonus"], 2.0)
        self.assertEqual(term_info["reward_components"]["terminal_on_time_bonus"], 5.0)

    def test_terminated_and_truncated_are_separate(self) -> None:
        env = self._make_env()
        observation, _ = env.reset(options={"episode_id": 1})
        action = self._pick_first_feasible_action(observation)
        _, _, terminated, truncated, _ = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

        trunc_env = self._make_env(max_consecutive_defers=1, invalid_action_limit=1)
        trunc_env.reset(options={"episode_id": 0})
        _, _, terminated, truncated, _ = trunc_env.step(999)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_env_dataset_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/rl_env_dataset.json.gz"
            save_env_dataset(self.dataset, Path(path))
            loaded = load_env_dataset(path)
            self.assertEqual(len(loaded.tasks), len(self.dataset.tasks))
            self.assertEqual(len(loaded.machines), len(self.dataset.machines))
            self.assertEqual(len(loaded.episodes), len(self.dataset.episodes))

    def test_behavior_validation_report_passes(self) -> None:
        report = validate_environment_behavior()
        self.assertTrue(report["summary"]["passed_all_checks"], msg=str(report))

    def test_sb3_check_env_when_dependencies_are_available(self) -> None:
        try:
            import gymnasium  # noqa: F401
            from stable_baselines3.common.env_checker import check_env
        except ModuleNotFoundError as exc:
            self.skipTest(f"SB3 environment check skipped because dependency is missing: {exc}")
            return

        if not GYMNASIUM_AVAILABLE:
            self.skipTest("Gymnasium compatibility module is active instead of the real gymnasium package.")

        env = self._make_env()
        check_env(env, warn=True)


if __name__ == "__main__":
    unittest.main()
