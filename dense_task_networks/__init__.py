from gym.envs.registration import register

register(
  id='dtn-v0',
  entry_point='dense_task_networks.envs:DTNEnv',
)
