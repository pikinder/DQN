import numpy as np
from dqn import VisualDQN, StateDQN
from util import preprocess_pong, preprocess_cartpole

pong_config = {
    'double_q': True,
    'double_q_freq': 10000,

    'game': 'Pong-v0',
    'frame': preprocess_pong,
    'q': VisualDQN,
    'q_params': {},
    'actions': 6,
    'state_dtype': np.uint8,
    'state_shape': [80, 80],
    'state_memory': 10 ** 6,
    'state_time': 4,

    'episodes': 10 ** 7,
    'episodes_validate': 5,
    'episodes_validate_runs': 2,
    'episodes_save_interval': 50,

    'batch_size': 32,

    'step_startrl': 5 * 10 ** 4,  #
    'step_eps_min': 1. / (10. ** 6.),
    'step_eps_mul': 1.,

    'eps_minval': .05,

    'gamma': 0.99,
}

acrobot_config = {
    'double_q': True,
    'double_q_freq': 1000,

    'game': 'Acrobot-v1',
    'frame': preprocess_cartpole,
    'q': StateDQN,
    'q_params': {},
    'actions': 3,
    'state_dtype': np.float32,
    'state_shape': [1, 6],
    'state_memory': 10 ** 4,
    'state_time': 1,

    'episodes': 10 ** 7,
    'episodes_validate': 2,
    'episodes_validate_runs': 10,
    'episodes_save_interval': 100,

    'batch_size': 128,

    'step_startrl': 5 * 10 ** 2,  #
    'step_eps_min': 1. / (10. ** 5.),
    'step_eps_mul': 1.,

    'eps_minval': .05,

    'gamma': 0.95,
}

cartpole_config = {
    'double_q': True,
    'double_q_freq': 1000,

    'game': 'CartPole-v1',
    'frame': preprocess_cartpole,
    'q': StateDQN,
    'q_params': {},
    'actions': 2,
    'state_dtype': np.float32,
    'state_shape': [1, 4],
    'state_memory': 10 ** 4,
    'state_time': 1,

    'episodes': 10 ** 7,
    'episodes_validate': 2,
    'episodes_validate_runs': 10,
    'episodes_save_interval': 100,

    'batch_size': 32,

    'step_startrl': 5 * 10 ** 2,  #
    'step_eps_min': 1. / (10. ** 4.),
    'step_eps_mul': 1.,

    'eps_minval': .05,

    'gamma': 0.99,
}
