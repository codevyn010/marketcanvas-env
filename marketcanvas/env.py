from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from marketcanvas.canvas import Canvas
from marketcanvas.actions import build_action_space, execute_action
from marketcanvas.observer import (
    PygameDisplay,
    build_observation_space,
    get_gymnasium_observation,
    render_to_array,
    render_to_png,
)
from marketcanvas.rewards import TargetSpec, compute_reward


DEFAULT_PROMPT = (
    "Create a Summer Sale email banner with a headline, "
    "a yellow CTA button, and good contrast"
)


class MarketCanvasEnv(gym.Env):
    """A deterministic 2D design canvas environment for RL training.

    The agent constructs a marketing asset on an 800x600 canvas.
    A composite reward is computed at episode termination based on
    constraint satisfaction, layout aesthetics, and WCAG compliance.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        target_prompt: str = DEFAULT_PROMPT,
        max_steps: int = 50,
    ):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.canvas = Canvas(width=800, height=600)
        self.target_prompt = target_prompt
        self.max_steps = max_steps

        self.observation_space = build_observation_space(self.canvas.width, self.canvas.height)
        self.action_space = build_action_space()

        self._target_spec: TargetSpec | None = None
        self._step_count = 0
        self._pygame_display: PygameDisplay | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        self.canvas.clear()
        self._step_count = 0

        prompt = self.target_prompt
        if options and "target_prompt" in options:
            prompt = options["target_prompt"]

        self._target_spec = TargetSpec.from_prompt(prompt)

        obs = get_gymnasium_observation(self.canvas, self._step_count)
        info = {"target_prompt": prompt, "target_spec": self._target_spec}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        self._step_count += 1

        action_result = execute_action(self.canvas, action)

        terminated = action_result.get("done", False)
        truncated = self._step_count >= self.max_steps

        is_terminal = terminated or truncated
        if is_terminal and self._target_spec is not None:
            reward_info = compute_reward(self.canvas, self._target_spec)
            reward = reward_info["reward"]
        else:
            reward = 0.0
            reward_info = {}

        obs = get_gymnasium_observation(self.canvas, self._step_count)
        info = {
            "action_result": action_result,
            "step": self._step_count,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return render_to_array(self.canvas)
        if self.render_mode == "human":
            if self._pygame_display is None:
                self._pygame_display = PygameDisplay(
                    self.canvas.width,
                    self.canvas.height,
                    fps=self.metadata["render_fps"],
                )
            self._pygame_display.render(self.canvas)
            return None
        return None

    def save_canvas(self, path: str) -> None:
        render_to_png(self.canvas, path)

    def close(self) -> None:
        if self._pygame_display is not None:
            self._pygame_display.close()
            self._pygame_display = None
