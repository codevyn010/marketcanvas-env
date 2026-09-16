import numpy as np
import pytest
import gymnasium as gym

import marketcanvas  # triggers env registration
from marketcanvas.actions import ActionType
from marketcanvas.env import MarketCanvasEnv


class TestEnvCreation:
    def test_make_env(self):
        env = gym.make("marketcanvas/MarketCanvas-v0")
        assert env is not None
        env.close()

    def test_reset_returns_valid_obs(self):
        env = MarketCanvasEnv()
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_deterministic(self):
        env1 = MarketCanvasEnv()
        env2 = MarketCanvasEnv()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1["elements"], obs2["elements"])
        env1.close()
        env2.close()


class TestEnvStep:
    def test_step_noop(self):
        env = MarketCanvasEnv()
        env.reset(seed=0)
        action = {
            "action_type": ActionType.NO_OP,
            "params": np.zeros(10, dtype=np.float32),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert reward == 0.0
        assert not terminated
        env.close()

    def test_step_add_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=0)
        params = np.array([0, 0.25, 0.1, 0.5, 0.1, 0, 0, 0, 0, 0.2], dtype=np.float32)
        action = {"action_type": ActionType.ADD_TEXT, "params": params}
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["action_result"]["success"]
        assert obs["element_count"] == 1
        env.close()

    def test_done_terminates(self):
        env = MarketCanvasEnv()
        env.reset(seed=0)
        action = {
            "action_type": ActionType.DONE,
            "params": np.zeros(10, dtype=np.float32),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated
        assert reward != 0.0
        env.close()

    def test_truncation_at_max_steps(self):
        env = MarketCanvasEnv(max_steps=3)
        env.reset(seed=0)
        noop = {
            "action_type": ActionType.NO_OP,
            "params": np.zeros(10, dtype=np.float32),
        }
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(noop)
            assert not truncated
        _, _, terminated, truncated, _ = env.step(noop)
        assert truncated
        env.close()


class TestEnvRender:
    def test_rgb_array(self):
        env = MarketCanvasEnv(render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (600, 800, 3)
        env.close()


class TestRandomRollouts:
    def test_100_random_episodes(self):
        env = MarketCanvasEnv()
        for ep in range(100):
            obs, info = env.reset(seed=ep)
            assert env.observation_space.contains(obs)
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
                if terminated or truncated:
                    break
        env.close()
