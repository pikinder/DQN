from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from dqn import StateDQN,AtariDQN
from util import preprocess_cartpole, preprocess_atari

breakout_config = {
    'double_q': True,
    'double_q_freq': 10000,

    'game': 'BreakoutDeterministic-v3',
    'frame': preprocess_atari,
    'q': AtariDQN,
    'q_params': {},
    'actions': 6,
    'state_dtype': np.uint8,
    'state_shape': [84, 84],
    'state_memory': 10 ** 6,
    'state_time': 4,

    'episodes': 10 ** 7,
    'episodes_validate': 100,
    'episodes_validate_runs': 2,
    'episodes_save_interval': 50,
    'tensorboard_interval':1000,

    'batch_size': 32,

    'step_startrl': 5 * 10 ** 4,  #
    'step_eps_min': 1. / (10. ** 6.),
    'step_eps_mul': 1.,

    'eps_minval': .1,

    'gamma': 0.99,
}


pong_config = {
    'double_q': True,
    'double_q_freq': 10000,

    'game': 'PongDeterministic-v3',
    'frame': preprocess_atari,
    'q': AtariDQN,
    'q_params': {},
    'actions': 6,
    'state_dtype': np.uint8,
    'state_shape': [84, 84],
    'state_memory': 10 ** 6,
    'state_time': 4,

    'episodes': 10 ** 7,
    'episodes_validate': 5,
    'episodes_validate_runs': 2,
    'episodes_save_interval': 50,
    'tensorboard_interval':1000,

    'batch_size': 32,

    'step_startrl': 5 * 10 ** 4,  #
    'step_eps_min': 1. / (10. ** 6.),
    'step_eps_mul': 1.,

    'eps_minval': .1,

    'gamma': 0.99,
}

cartpole_config = {
    'double_q': True,
    'double_q_freq': 1000,

    'game': 'CartPole-v0',
    'frame': preprocess_cartpole,
    'q': StateDQN,
    'q_params': {},
    'actions': 2,
    'state_dtype': np.float32,
    'state_shape': [1, 4],
    'state_memory': 10 ** 3,
    'state_time': 1,

    'episodes': 10 ** 7,
    'episodes_validate': 2,
    'episodes_validate_runs': 10,
    'episodes_save_interval': 1,
    'tensorboard_interval':1,
    'batch_size': 32,

    'step_startrl': 5 * 10 ** 2,  #
    'step_eps_min': 1. / (10. ** 3.),
    'step_eps_mul': 1.,

    'eps_minval': .1,

    'gamma': 0.99,
}
