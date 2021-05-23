import math

import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding


class SingleDTNEnv(gym.GoalEnv):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50
  }

  MOVE_DIRECTION = [[0, -1], [1, 0], [0, 1], [-1, 0]]  # up, right, down, left

  def __init__(self):
    self.grid_width = 20
    self.grid_height = 5
    self.start_position = [1, 2]
    self.goal_position = [18, 2]
    self.agent_position = self.start_position

    self.max_steps = 30

    self.game_over = False

    self.action_space = spaces.Discrete(4)  # 4 directions

    self.observe_as_xy_coords = True

    if self.observe_as_xy_coords:
      observation_space = spaces.Box(
        low=0,
        high=max(self.grid_width, self.grid_height),
        shape=(2,),
        dtype='uint8'
      )
    else:
      observation_space = spaces.Box(
        low=0,
        high=1,
        shape=(self.grid_width, self.grid_height, 1),
        dtype='uint8'
      )

    self.observation_space = spaces.Dict({
      'observation': observation_space,
      'desired_goal': observation_space,
      'achieved_goal': observation_space
    })

    self.seed()
    self.viewer = None
    self.state = None

    self.step_num = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _get_new_position(self, action):
    """
    Apply the action to change the agent's position

    :param action:
    :return:
    """
    new_agent_pos = [self.agent_position[0] + self.MOVE_DIRECTION[action][0],
                     self.agent_position[1] + self.MOVE_DIRECTION[action][1]]

    # Check if the new location is out of boundary
    if new_agent_pos[0] < 0 or new_agent_pos[1] < 0 \
            or new_agent_pos[0] > (self.grid_width - 1) or new_agent_pos[1] > (self.grid_height - 1):
      return self.agent_position
    else:
      return new_agent_pos

  def _take_action(self, action):
    """
    Performs the action on the grid. Updates the agent's position.

    :param action:
    :return:
    """
    # Move the agent in that direction
    new_agent_pos = self._get_new_position(action)
    self.agent_position = new_agent_pos

    return

  def _get_state(self, obs_xy):
    """
    Get the grid information.

    :return:
    """
    if obs_xy:
      # Just return the (x,y) coordinate of the agent
      state = np.array(self.agent_position)
    else:
      raise NotImplementedError()

    return state

  def _get_obs(self):
    """
    Return the observation as a dictionary of observation and goals

    :return:
    """
    if self.observe_as_xy_coords:
      desired_goal = np.array(self.goal_position)
    else:
      raise NotImplementedError()

    obs = {
      'observation': self._get_state(self.observe_as_xy_coords),
      'desired_goal': desired_goal,
      'achieved_goal': self._get_state(self.observe_as_xy_coords)
    }
    return obs

  def step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    self._take_action(action)
    obs = self._get_obs()
    info = {}
    reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

    self.step_num += 1

    # Episode is done when the agent reaches the goal state or runs out of time
    if reward == 0.0 or self.step_num == self.max_steps or self.game_over:
      done = 1
    else:
      done = 0

    if done: info['TimeLimit.truncated'] = True
    info['is_success'] = np.allclose(reward, 0.)

    return obs, reward, done, info

  def compute_reward(self, achieved_goal, desired_goal, info):
    """Compute the step reward. This externalizes the reward function and makes
    it dependent on an a desired goal and the one that was achieved. If you wish to include
    additional rewards that are independent of the goal, you can include the necessary values
    to derive it in info and compute it accordingly.
    Args:
        achieved_goal (object): the goal that was achieved during execution
        desired_goal (object): the desired goal that we asked the agent to attempt to achieve
        info (dict): an info dictionary with additional information
    Returns:
        float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
        goal. Note that the following should always hold true:
            ob, reward, done, info = env.step()
            assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
    """
    assert (achieved_goal.shape == desired_goal.shape)
    # Compare if the dimensions 1 to N all match, and return vector of the first dimension (i.e. batch)
    # Check if computing reward for vectorized environment or a single environment
    # Note: Reward is 0 if the agent achieves the goal, and -1 otherwise
    if (achieved_goal == desired_goal).all():
      return 0.0
    else:
      return -1.0

  def reset(self):
    self.game_over = False
    self.step_num = 0

    self.agent_position = self.start_position

    return self._get_obs()

  def render(self, mode='human'):
    # print("Agent position: " + str(self.agent_position) + "; goal position: " + str(self.goal_position) + ".")

    pass

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
