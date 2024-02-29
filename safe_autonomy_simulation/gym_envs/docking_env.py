"""An example gymnasium environment built using the DockingSimulator"""


from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from safe_autonomy_simulation.docking_simulator import DockingSimulator
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft


class DockingEnv(gym.Env):
    def __init__(self, chief_init: dict, deputy_init: dict) -> None:
        # initialize simulator with chief and deputy spacecraft
        self.simulator = DockingSimulator(
            entities={
                "chief": CWHSpacecraft(name="chief", **chief_init),
                "deputy": CWHSpacecraft(name="deputy", **deputy_init), 
            },
            frame_rate=1,
        )

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
        # set reward as 1 / dist(chief, deputy)
        chief_pos = obs["chief"][:3]
        dep_pos = obs["deputy"][:3]
        dist = np.linalg.norm(chief_pos, dep_pos)
        return 1 / dist
    
    def _get_terminated(self, obs):
        # terminate episode when dist(chief, deputy) < 5
        chief_pos = obs["chief"][:3]
        dep_pos = obs["deputy"][:3]
        dist = np.linalg.norm(chief_pos, dep_pos) 
        return dist < 5
