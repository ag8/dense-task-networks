import math

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control.rendering import Color
from gym.utils import seeding


class Robot2dEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
  }

  def __init__(self, goal_velocity=0):
    self.step_num = 0
    self.max_steps = 50000

    self.min_effector_action = -1.0
    self.max_effector_action = 1.0

    self.min_grab_action = -1.0
    self.max_grab_action = 1.0

    self.min_board_position = -10
    self.max_board_position = 10
    self.power = 0.1
    self.max_speed = 0.7

    self.goal_square_center = (0, 0)
    self.goal_square_size = 2.0

    self.left_block_start = (-8, 0)
    self.left_block_style = 'C'

    self.right_block_start = (8, 0)
    self.right_block_style = 'O'

    self.style_min = -1  # C
    self.style_max = 1  # O

    # The action is a combination of two things:
    # 1. which direction to move the effector in, ranging from [-1, -1] to [1, 1].
    # 2. whether to grab the underlying obt, ranging from 0 to 1.

    # Here, the first element refers to left-right motion, the second element refers to up-down motion,
    # and the last element refers to the grabbing motion.
    self.low_action = np.array(
      [self.min_effector_action, self.min_effector_action, self.min_grab_action], dtype=np.float32
    )

    self.high_action = np.array(
      [self.max_effector_action, self.max_effector_action, self.max_grab_action], dtype=np.float32
    )

    # The board state is a combination of six things:
    # 1. The position of the end effector  (2 thingeroonis)
    # 2. The position of the left block    (2 thingeroonis)
    # 3. The style of the left block       (1 thingerooni)
    # 4. The position of the right block   (2 thingeroonis)
    # 5. The style of the right block      (1 thingerooni)
    # 6. Whether we're currently grabbing  (1 thingerooni)
    self.low_state = np.array(
      [self.min_board_position, self.min_board_position,  # 1
       self.min_board_position, self.min_board_position,  # 2
       self.style_min,  # 3
       self.min_board_position, self.min_board_position,  # 4
       self.style_min,  # 5
       self.min_grab_action  # 6
       ], dtype=np.float32
    )
    self.high_state = np.array(
      [self.max_board_position, self.max_board_position,  # 1
       self.max_board_position, self.max_board_position,  # 2
       self.style_max,  # 3
       self.max_board_position, self.max_board_position,  # 4
       self.style_max,  # 5
       self.min_grab_action,  # 6
       ], dtype=np.float32
    )

    self.viewer = None

    self.action_space = spaces.Box(
      low=self.low_action,
      high=self.high_action,
      dtype=np.float32
    )

    self.observation_space = spaces.Box(
      low=self.low_state,
      high=self.high_state,
      dtype=np.float32
    )

    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _determine_blocks_grabbed(self, effector_position, left_block_position, right_block_position):
    """
    BUGBUG: Make this detect intesections with the actual appropriate shapes of the blocks
    (i.e., not just squares)
    """
    if left_block_position[0] - 0.5 < effector_position[0] < left_block_position[0] + 0.5 and \
            left_block_position[1] - 0.5 < effector_position[1] < left_block_position[1] + 0.5:
      return 0

    if right_block_position[0] - 0.5 < effector_position[0] < right_block_position[0] + 0.5 and \
            right_block_position[1] - 0.5 < effector_position[1] < right_block_position[1] + 0.5:
      return 1

    return -1

  def step(self, action):
    effector_position = [self.state[0], self.state[1]]
    left_block_position = [self.state[2], self.state[3]]
    right_block_position = [self.state[5], self.state[6]]

    effector_action = np.array([action[0], action[1]])
    grab = action[2] > 0.0  # we consider the effector to be grabbing if the last action part is >0.5.

    # Determine which, if any, blocks we're grabbing onto
    block_grabbed = self._determine_blocks_grabbed(effector_position, left_block_position, right_block_position)

    # Determine the velocity of the end effector
    velocity = self.power * effector_action

    # Clamp it if necessary
    if velocity[0] > self.max_speed:
      velocity[0] = self.max_speed
    if velocity[1] > self.max_speed:
      velocity[1] = self.max_speed

    # Change the end effector's position accordingly
    effector_position += velocity

    # Make sure the effector's position is still within the board
    if effector_position[0] > self.max_board_position:
      effector_position[0] = self.max_board_position
    if effector_position[0] < self.min_board_position:
      effector_position[0] = self.min_board_position
    if effector_position[1] > self.max_board_position:
      effector_position[1] = self.max_board_position
    if effector_position[1] < self.min_board_position:
      effector_position[1] = self.min_board_position

    # Change the position of any block that was grabbed onto as well
    if block_grabbed == 0:
      left_block_position = effector_position
    if block_grabbed == 1:
      right_block_position = effector_position

    # We're done if both blocks' centers of mass (bugbug make this) are in the goal square,
    # and we're not grabbing onto anything (e.g. to hold it up).
    done = bool(
      left_block_position[0] > self.goal_square_center[0] - self.goal_square_size / 2 and \
      left_block_position[0] < self.goal_square_center[0] + self.goal_square_size / 2 and \
      left_block_position[1] > self.goal_square_center[1] - self.goal_square_size / 2 and \
      left_block_position[1] < self.goal_square_center[1] + self.goal_square_size / 2 and \
      right_block_position[0] > self.goal_square_center[0] - self.goal_square_size / 2 and \
      right_block_position[0] < self.goal_square_center[0] + self.goal_square_size / 2 and \
      right_block_position[1] > self.goal_square_center[1] - self.goal_square_size / 2 and \
      right_block_position[1] < self.goal_square_center[1] + self.goal_square_size / 2 and \
      not grab
    )

    # FOR NOW, make the goal depend only on the left block (BUGBUG temporary)
    # and make sure it's not taking way too long
    done = bool(
      left_block_position[0] > self.goal_square_center[0] - self.goal_square_size / 2 and \
      left_block_position[0] < self.goal_square_center[0] + self.goal_square_size / 2 and \
      left_block_position[1] > self.goal_square_center[1] - self.goal_square_size / 2 and \
      left_block_position[1] < self.goal_square_center[1] + self.goal_square_size / 2 and \
      not grab or \
      self.step_num > self.max_steps
    )

    reward = 0
    if done:
      reward = 100.0
    # reward -= math.pow(action[0], 2) * 0.1
    # reward -= math.pow(action[1], 2) * 0.1
    reward -= 1

    self.state = np.array([effector_position[0], effector_position[1],
                           left_block_position[0], left_block_position[1],
                           -1.0 if self.left_block_style == "C" else 1.0,
                           right_block_position[0], right_block_position[1],
                           1.0 if self.right_block_style == "C" else 1.0,
                           action[2]])

    # print("Reward: " + str(reward))

    return self.state, reward, done, {}

  def reset(self):
    self.state = np.array([  # self.np_random.uniform(low=-5, high=5), self.np_random.uniform(low=-5, high=5),  # 1
      -7, 0,  # bugbug for now have it start near the first block so it learns something
      self.left_block_start[0], self.left_block_start[1],  # 2
      -1.0 if self.left_block_style == "C" else 1.0,  # 3
      self.right_block_start[0], self.right_block_start[1],  # 4
      -1.0 if self.right_block_style == "C" else 1.0,  # 5
      -1.0  # 6
    ])

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
