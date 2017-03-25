from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


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
        state = self.states.take(np.r_[-state_length+1:1]+self.idx,axis=0,mode='wrap')
        return state.transpose(1,2,0)

    def sample_experience(self, batch_size,state_length):
        """
        Sample experience.

        Warning, because of the optimisation that the states are stored in order, the next state when sampling is only valid.
        If the previous state did not result in ending the epsiode. (Signalled by done)

        TODO: Rewrite this function to make it the sampling fully correct. Or verify the speed that the current one is much faster....

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