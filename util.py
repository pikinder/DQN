import os
import time

import numpy as np


def get_log_dir(log_base, suffix):
    """
    For each run create a direction
    :param log_base:
    :param suffix:
    :return:
    """
    run = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = '%s/%s_%s' % (log_base, run, suffix)
    os.mkdir(log_dir)
    return log_dir

class Experience(object):
    """
    Class to store the experience replay memory
    """
    def __init__(self, memory_size, state_shape,dtype):
        self.memory_size = memory_size
        self.states = np.zeros((memory_size,) + tuple(state_shape),dtype=dtype)
        self.actions = np.zeros(memory_size).astype(np.int32)
        self.done = np.zeros(memory_size)
        self.rewards = np.zeros(memory_size)
        self.count = 0
        self.idx = 0

    def add(self, state, action, reward, done):
        """

        :param state:
        :param action:
        :param reward:
        :param done:
        :return:
        """
        self.count = min(self.memory_size,self.count+1)
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = done

        self.idx = (self.idx+1)%self.memory_size

    def can_sample(self,batch_size,state_length):
        return self.count>batch_size+state_length

    def get_last_state(self,state_length):
        idx = np.r_[-state_length+1:1]+self.idx
        return self.states[idx].transpose(1,2,0)

    def sample_experience(self, batch_size,state_length):
        """
        Sample experience.

        Warning, because of the optimisation that the states are stored in order, the next state when sampling is only valid
        if the previous state did not result in ending the epsiode. (Signalled by done)
        :param batch_size:
        :return:
        """
        if not self.can_sample(batch_size,state_length):
            raise RuntimeError('Not enough experience....')

        # Line below can be optimised I guess...
        idx = np.random.permutation(self.count - state_length-1)[:batch_size]
        sample_states =  np.zeros((batch_size,) + tuple(self.states.shape[1:])+(state_length,))
        next_states =  np.zeros((batch_size,) + tuple(self.states.shape[1:])+(state_length,))
        for s,id in enumerate(idx):
            sample_states[s] = self.states[id:id+state_length].transpose(1,2,0)
            next_states[s] = self.states[id+1:id+1 + state_length].transpose(1, 2, 0)

        return sample_states, self.actions[idx+state_length-1], self.rewards[idx+state_length-1], self.done[idx+state_length-1], next_states


def preprocess_cartpole(frame):
    return frame[np.newaxis,:,np.newaxis]


def preprocess_pong(frame):
    """
    Preprocess the image as suggested on kharpathys blog
    :param state:
    :return:
    """
    frame = frame.copy()[35:195,:,0]
    frame = frame[::2,::2]
    frame[frame == 144] = 0 # erase background (background type 1)
    frame[frame == 109] = 0 # erase background (background type 2)
    frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
    return frame[:,:,np.newaxis]