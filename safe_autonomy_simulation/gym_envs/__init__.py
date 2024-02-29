from gymnasium.envs.registration import register

register(
     id="DockingEnv-v0",
     entry_point="safe_autonomy_simulation.gym_envs:DockingEnv",
     max_episode_steps=300,
)

register(
     id="InspectionEnv-v0",
     entry_point="safe_autonomy_simulation.gym_envs:InspectionEnv",
     max_episode_steps=300,
)