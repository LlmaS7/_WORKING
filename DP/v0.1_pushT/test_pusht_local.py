"""Targeted checks for episode windows and the adapted environment.

Run: E:\\_WORKING\\.venv\\Scripts\\python.exe -m unittest test_pusht_local -v
"""
import unittest

import numpy as np
import torch

from run_pusht import (PushTSequences, dataset_keypoint_map, fit_stats, make_env, normalize,
                       split_episodes, unnormalize)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.repeat(np.arange(40, dtype=np.float32)[:, None], 20, axis=1)
        self.action = self.obs[:, :2].copy()
        self.ends = np.array([20, 40])

    def test_windows_never_cross_episodes(self):
        ds = PushTSequences(self.obs, self.action, self.ends, [0, 1])
        for i in range(len(ds)):
            _, action = ds[i]
            self.assertEqual(bool((action < 20).all()), bool(action[0, 0] < 20))
        first_obs, first_action = ds[0]
        torch.testing.assert_close(first_obs[:, 0], torch.tensor([0., 0.]))
        self.assertEqual(first_action[1, 0].item(), 0.)
        _, last_action = ds[len(ds) - 1]
        self.assertEqual(last_action[-1, 0].item(), 39.)

    def test_normalizer_uses_only_training_episodes(self):
        stats = fit_stats(self.obs, self.action, self.ends, [0])
        self.assertEqual(stats["obs_min"][0].item(), 0.)
        self.assertEqual(stats["obs_range"][0].item(), 19.)
        values = torch.tensor(self.action)
        torch.testing.assert_close(unnormalize(normalize(values, "action", stats), "action", stats), values)

    def test_episode_split_is_disjoint_and_reproducible(self):
        train, val = split_episodes(np.arange(1, 11) * 20, 42)
        train2, val2 = split_episodes(np.arange(1, 11) * 20, 42)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(set(train) | set(val)), 10)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(val, val2)

    def test_keypoint_calibration_ignores_a_pose_outlier(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(9, 2)) * 30
        state = rng.uniform(0, 300, size=(50, 5))
        state[:, 4] = np.linspace(0, 2 * np.pi, 50)
        observations = []
        for row in state:
            c, s = np.cos(row[4]), np.sin(row[4])
            world = points @ np.array([[c, s], [-s, c]]) + row[2:4]
            observations.append(np.r_[world.flatten(), row[:2]])
        observations = np.array(observations)
        observations[0, :18] += 2
        recovered = dataset_keypoint_map(observations, state)
        np.testing.assert_allclose(recovered["block"], points, atol=1e-8)


class EnvironmentTests(unittest.TestCase):
    def test_contact_callback_and_pushing(self):
        env = make_env(42)
        try:
            env.reset_to_state = np.array([256., 440., 256., 300., 0.])
            env.reset()
            before = np.array(env.block.position)
            contacts = 0
            for _ in range(20):
                _, _, _, info = env.step(np.array([256., 250.]))
                contacts = max(contacts, info["n_contacts"])
            self.assertGreater(contacts, 0)
            self.assertGreater(np.linalg.norm(np.array(env.block.position) - before), 1.)
        finally:
            env.close()

    def test_reset_step_and_render_are_reproducible(self):
        env = make_env(42)
        try:
            before = env.reset()
            action = before[18:20] + [10, -10]
            after, reward, done, _ = env.step(action)
            self.assertTrue(np.isfinite(after).all())
            self.assertGreaterEqual(float(reward), 0.)
            self.assertLessEqual(float(reward), 1.)
            self.assertEqual(env.render("rgb_array").shape, (256, 256, 3))
            env.seed(42)
            np.testing.assert_allclose(env.reset(), before)
            after2, reward2, done2, _ = env.step(action)
            np.testing.assert_allclose(after2, after)
            self.assertEqual(reward2, reward)
            self.assertEqual(done2, done)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
