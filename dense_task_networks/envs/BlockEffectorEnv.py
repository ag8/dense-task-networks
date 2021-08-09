import math

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control.rendering import Color
from gym.utils import seeding


def parse_file(world_file):
  with open(world_file) as f:
    lines = f.read().splitlines()

  # The number of lines is the height
  h = len(lines)

  # The length of each line is the width
  w = len(lines[0])

  pos_dict = {}
  
  effector_position = (0, 0)

  for row in range(h):
    for col in range(w):
      if lines[row][col].isdigit():
        pos_dict[int(lines[row][col])] = (row, col)
      elif lines[row][col] == 'E':
        effector_position = (row, col)
  
  # pos_dict = sorted(pos_dict)
  
  return pos_dict, effector_position, w, h


def _vector_from_action(action):
  if action == 0:  # left
    return [0, -1]
  elif action == 1:  # up
    return [-1, 0]
  elif action == 2:  # right
    return [0, 1]
  elif action == 3:  # down
    return [1, 0]
  else:
    return [0, 0]


class BlockEffectorEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
  }

  def __init__(self, world_file='/home/andrew/genrl/envs/dense_task_networks/dense_task_networks/envs/rooms/basic_switch.txt', n_noops=0):
    self.step_num = 0
    self.max_steps = 50

    self.block_starts, self.effector_start, self.w, self.h = parse_file(world_file)
    self.grabbing = False

    self.block_goals = {0: [0, 9], 1: [9, 0]}

    self.viewer = None

    self.action_space = spaces.Discrete(6 + n_noops)

    self.low_state = np.array(
      [0, 0,
       0, 0,
       0, 0,
       0
       ], dtype=np.int
    )
    self.high_state = np.array(
      [self.h, self.w,
       self.h, self.w,
       self.h, self.w,
       1
       ], dtype=np.int
    )

    self.viewer = None

    self.observation_space = spaces.Box(
      low=self.low_state,
      high=self.high_state,
      dtype=np.int
    )

    # self.observation_space = spaces.Tuple((  # todo: figure out how to generalize this to n blocks
    #             spaces.Discrete(self.h),  # effector row
    #             spaces.Discrete(self.w),  # effector col
    #             spaces.Discrete(self.h),  # block 1 row
    #             spaces.Discrete(self.w),  # block 1 col
    #             spaces.Discrete(self.h),  # block 2 row
    #             spaces.Discrete(self.w),   # block 2 col
    #             spaces.Discrete(2)  # grabbing
    # ))

    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _clamp_to_field(self, pos):
    if pos[0] < 0:
      pos[0] = 0
    if pos[1] < 0:
      pos[1] = 0
    if pos[0] > self.h:
      pos[0] = self.h
    if pos[1] > self.w:
      pos[1] = self.w

    return pos

  def _attempt_move(self, direction_arr, position):
    new_position = [position[0] + direction_arr[0], position[1] + direction_arr[1]]

    return self._clamp_to_field(new_position)
  
  def _move(self, direction_arr, effector_position, block_positions):
    new_effector_position = self._attempt_move(direction_arr, effector_position)

    if self.grabbing:
      # figure out wch block we're grabbing onto
      for i in range(len(block_positions)):
        if block_positions[i][0] == effector_position[0] and block_positions[i][1] == effector_position[1]:
          block_positions[i] = new_effector_position

    return new_effector_position, block_positions
      

  def step(self, action):
    self.step_num += 1

    effector_position = [self.state[0], self.state[1]]
    
    block_positions = [(self.state[i], self.state[i + 1]) for i in range(2, len(self.state) - 2, 2)]


    if action == 5:
      self.grabbing = True
    elif action == 6:
      self.grabbing = False
    elif action > 6:  # noop
      pass
    else:
      effector_position, block_positions = self._move(_vector_from_action(action), effector_position, block_positions)


    done = all([block_positions[i] == self.block_goals[i] for i in range((len(self.state) - 3) // 2)]) and not self.grabbing or self.step_num > self.max_steps


    reward = 0
    if done and self.step_num <= self.max_steps:
      reward = 1.0
    reward -= 1

    self.state = np.array([effector_position[0], effector_position[1],
                           block_positions[0][0], block_positions[0][1],
                           block_positions[1][0], block_positions[1][1],
                           1 if self.grabbing else 0])

    # print("Reward: " + str(reward))

    return self.state, reward, done, {}

  def reset(self):
    self.state = np.array([self.effector_start[0], self.effector_start[1],
                           self.block_starts[0][0], self.block_starts[0][1],
                           self.block_starts[1][0], self.block_starts[1][1],
                           1 if self.grabbing else 0])

    self.step_num = 1

    return np.array(self.state)

  def render(self, mode='human'):
    screen_width = 600
    screen_height = 600

    world_width = self.max_board_position - self.min_board_position
    scale = screen_width / world_width
    effectorradius = 10

    from gym.envs.classic_control import rendering

    if self.viewer is None:
      self.viewer = rendering.Viewer(screen_width, screen_height)

      # ltop = (- 0.5, - 0.5)
      # rtop = (+ 0.5, - 0.5)
      # lbot = (- 0.5, + 0.5)
      # rbot = (+ 0.5, + 0.5)
      # lblock = rendering.FilledPolygon([ltop, rtop, lbot, rbot])
      # lblock.set_color(0, 0, 1)
      # lblock.add_attr(rendering.Transform(translation=(0, 0)))
      # self.lblocktrans = rendering.Transform()
      # lblock.add_attr(self.lblocktrans)
      # self.viewer.add_geom(lblock)

    pos = [self.state[0], self.state[1]]
    grab = self.state[8]

    # Draw the target area
    gt = rendering.Transform(translation=((self.goal_square_center[0] - self.min_board_position) * scale,
                                          (self.goal_square_center[1] - self.min_board_position) * scale))
    _ = self.viewer.draw_polygon(((0.5 * self.goal_square_size * scale, 0.5 * self.goal_square_size * scale),
                                  (-0.5 * self.goal_square_size * scale, 0.5 * self.goal_square_size * scale),
                                  (-0.5 * self.goal_square_size * scale, -0.5 * self.goal_square_size * scale),
                                  (0.5 * self.goal_square_size * scale, -0.5 * self.goal_square_size * scale)),
                                 filled=True,
                                 color=(1, 0.6, 0.6)).add_attr(gt)

    # Draw the left block (not very efficiently)
    lbt = rendering.Transform(translation=(0, 0))
    lblock = self.viewer.draw_polygon(((0.5 * scale, 0.5 * scale), (-0.5 * scale, 0.5 * scale),
                                       (-0.5 * scale, -0.5 * scale), (0.5 * scale, -0.5 * scale)), filled=True,
                                      color=(0, 0, 1)).add_attr(lbt)
    lblock_pos = [self.state[2], self.state[3]]
    lbt.set_translation(
      (lblock_pos[0] - self.min_board_position) * scale, (lblock_pos[1] - self.min_board_position) * scale
    )

    # Draw the right block (not very efficiently)
    rbt = rendering.Transform(translation=(0, 0))
    rblock = self.viewer.draw_polygon(((0.5 * scale, 0.5 * scale), (-0.5 * scale, 0.5 * scale),
                                       (-0.5 * scale, -0.5 * scale), (0.5 * scale, -0.5 * scale)), filled=True,
                                      color=(0, 1, 1)).add_attr(rbt)
    rblock_pos = [self.state[5], self.state[6]]
    rbt.set_translation(
      (rblock_pos[0] - self.min_board_position) * scale, (rblock_pos[1] - self.min_board_position) * scale
    )

    # Draw the end effector
    t = rendering.Transform(translation=(0, 0))
    effectorcolor = (.5, 1, .5) if grab > 0 else (1, .5, .5)
    effector = self.viewer.draw_circle(radius=effectorradius, res=30, color=effectorcolor).add_attr(t)

    t.set_translation(
      (pos[0] - self.min_board_position) * scale, (pos[1] - self.min_board_position) * scale
    )

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

