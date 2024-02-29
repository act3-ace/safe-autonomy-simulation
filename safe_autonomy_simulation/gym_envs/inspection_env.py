"""
An example translational inspection gymnasium environment built
using the InspectionSimulator
"""


from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from safe_autonomy_simulation.inspection.inspection_simulator import InspectionSimulator


class InspectionEnv(gym.Env):
    def __init__(self, config: dict) -> None:
        # assume point model spacecraft (translational inspection)
        self.simulator = InspectionSimulator(**config)

        # each spacecraft obs = [x, y, z, xdot, ydot, zdot]
        self.observation_space = spaces.Dict({
            "chief": spaces.Box(-np.inf, np.inf, shape=(6,)),
            "deputy": spaces.Box(-np.inf, np.inf, shape=(6,)),
        })

        # each spacecraft is controlled by [xdot, ydot, zdot]
        self.action_space = spaces.Dict({
            "deputy": spaces.Box(-1, 1, shape=(3,))  # only the deputy is controlled
        })

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.simulator.reset()
        obs, info = self._get_obs(), self._get_info()
        return obs, info
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # update simulator state
        self.simulator.add_controls(action)
        self.simulator.step()

        # get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        terminated = self._get_terminated()
        truncated = False  # used to signal episode ended unexpectedly
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs = self.simulator.info
        return obs
    
    def _get_info(self):
        pass

    def _get_reward(self, obs):
        # set reward as number of inspected points
        reward = 0
        for inspector, points in self.simulator.inspection_points_map:
            reward += points.get_num_points_inspected()
            break  # assuming one inspector
        return reward
    
    def _get_terminated(self, obs):
        # terminate episode when 95% of points inspected
        percent_inspected = 0
        for inspector, points in self.simulator.inspection_points_map:
            percent_inspected = points.get_percentage_of_points_inspected()
            break  # assuming one inspector
        return percent_inspected > 0.95
