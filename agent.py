from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym
from replay import Experience


class QAgent(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir

        self.env = gym.make(config['game'])

        self.replay_memory = Experience(
            memory_size=config['state_memory'],
            state_shape=config['state_shape'],
            dtype=config['state_dtype']
        )

        self.net = config['q'](
            batch_size=config['batch_size'],
            state_shape=config['state_shape']+[config['state_time']],
            num_actions=config['actions'],
            summaries=True,
            **config['q_params']
        )

        # Disable the target network if needed
        if not self.config['double_q']:
            self.net.t_batch = self.net.q_batch

        with tf.variable_scope('RL_summary'):
            self.episode = tf.Variable(0.,name='episode')
            self.training_reward = tf.Variable(0.,name='training_reward')
            self.validation_reward = tf.Variable(0.,name='validation_reward')
            tf.summary.scalar(name='training_reward',tensor=self.training_reward)
            tf.summary.scalar(name='validation_reward',tensor=self.validation_reward)

        # Create tensorflow variables
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.update_target_network()

        self.summaries = tf.summary.merge_all()
        if log_dir is not None:
            self.train_writer = tf.summary.FileWriter(self.log_dir, self.session.graph)

        self.epsilon = 1.0
        self.steps = 0

    def update_target_network(self):
        """
        Update the parameters of the target network

        :return:
        """
        if self.config['double_q']:
            self.session.run(self.net.assign_op)

    def _update_training_reward(self,reward):
        """
        set the value of the training reward.
        This ensures it is stored and visualised on the tensorboard
        """
        self.session.run(self.training_reward.assign(reward))

    def _update_validation_reward(self,reward):
        """
        set the value of the validation reward.
        This ensures it is stored and visualised on the tensorboard
        """
        self.session.run(self.validation_reward.assign(reward))

    def get_training_state(self):
        """
        Get the last state
        :return:
        """
        return self.replay_memory.get_last_state(self.config['state_time'])

    def sample_action(self,state,epsilon):
        """
        Sample an action for the state according to the epsilon greedy strategy

        :param state:
        :param epsilon:
        :return:
        """
        if np.random.rand() <= epsilon:
            return np.random.randint(0,self.config['actions'])
        else:
            return self.session.run(self.net.q, feed_dict={self.net.x: state[np.newaxis].astype(np.float32)})[0].argmax()

    def update_state(self, old_state, new_frame):
        """

        :param old_state:
        :param new_frame:
        :return:
        """
        return np.concatenate([
            old_state[:, :, 1:],
            new_frame
        ], axis=2)

    def reset_to_zero_state(self):
        """
        Reset the state history to zeros and reset the environment to fill the first state
        :return:
        """
        return np.concatenate([
                    np.zeros(self.config['state_shape']+[self.config['state_time']-1]),
                    self.config['frame'](self.env.reset())
                ],axis=2)


    def update_epsilon_and_steps(self):
        if self.steps > self.config['step_startrl']:
            self.epsilon = max(self.config['eps_minval'],self.epsilon*self.config['step_eps_mul']-self.config['step_eps_min'])
        self.steps += 1


    def train_episode(self):
        # Load the last state and add a reset
        state = self.update_state(
            self.get_training_state(),
            self.config['frame'](self.env.reset())
        )

        # Store the starting state in memory
        self.replay_memory.add(
            state = state[:,:,-1],
            action = np.random.randint(self.config['actions']),
            reward = 0.,
            done = False
        )

        # Use flags to signal the end of the episode and for pressing the start button
        done = False
        press_fire = True
        ale_lives = None
        total_reward = 0
        while not done:
            if press_fire: # start the episode
                press_fire = False
                new_frame,reward,done, info = self.act(state,-1,True)
                if info.has_key('ale.lives'):
                    ale_lives = info['ale.lives']
            else:
                self.update_epsilon_and_steps()
                new_frame,reward,done, info = self.act(state,self.epsilon,True,ale_lives)
            state = self.update_state(state, new_frame)
            total_reward += reward

            if self.steps > self.config['step_startrl']:
                summaries,_ = self.train_batch()
                if self.steps % self.config['tensorboard_interval'] == 0:
                    self.train_writer.add_summary(summaries, global_step=self.steps)
                if self.steps % self.config['double_q_freq'] == 0.:
                    print("double q swap")
                    self.update_target_network()

        return total_reward


    def validate_episode(self,epsilon,visualise=False):
        state = self.reset_to_zero_state()
        done = False
        press_fire = True
        total_reward = 0.
        while not done:
            if press_fire:
                press_fire = False
                new_frame, reward, done,_ = self.act(state=state, epsilon=-1, store=False)
            else:
                new_frame, reward, done,_ = self.act(state=state, epsilon=epsilon, store=False)
            state = self.update_state(old_state=state, new_frame=new_frame)
            total_reward += reward
            if visualise:
                self.env.render()
        return total_reward

    def act(self, state, epsilon, store=False, ale_lives=None):
        """
        Perform an action in the environment.

        If it is an atari game and there are lives, it will end the episode in the replay memory if a life is lost.
        This is important in games lik

        :param epsilon: the epsilon for the epsilon-greedy strategy. If epsilon is -1, the no-op will be used in the atari games.
        :param state: the state for which to compute the action
        :param store: if true, the state is added to the replay memory
        :return: the observed state (processed), the reward and whether the state is final
        """
        if epsilon == -1:
            action = 1
        else:
            action = self.sample_action(state=state,epsilon=epsilon)
        raw_frame, reward, done, info = self.env.step(action)

        # Clip rewards to -1,0,1
        reward = np.sign(reward)

        # Preprocess the output state
        new_frame = self.config['frame'](raw_frame)

        # If needed, store the last frame
        if store:
            # End episodes if lives are involved in the atari game.
            # This is important in e.g. breakout
            # If this is not included, dropping the ball is not penalised.
            # By marking the end of the reward propagation, the maximum reward is limited
            # This makes learning faster.
            store_done = done
            if ale_lives is not None and info.has_key('ale.lives') and info['ale.lives']<ale_lives:
                store_done = True
            self.replay_memory.add(state[:,:,-1],action,reward,store_done)
        return new_frame, reward, done, info


    def train_batch(self):
        """
        Sample a batch of training samples from the replay memory.
        Compute the target Q values
        Perform one SGD update step

        :return: summaries, step
        summaries: the tensorflow summaries that can be put into a log.
        step, the global step from tensorflow. This represents the number of updates
        """

        # Sample experience
        xp_states, xp_actions, xp_rewards, xp_done, xp_next = self.replay_memory.sample_experience(
            self.config['batch_size'],
            self.config['state_time']
        )

        # Create a mask on which output to update
        q_mask = np.zeros((self.config['batch_size'], self.config['actions']))
        for idx, a in enumerate(xp_actions):
            q_mask[idx, a] = 1

        # Use the target network to value the next states
        next_actions, next_values = self.session.run(
            [self.net.q_batch,self.net.t_batch],
            feed_dict={self.net.x_batch: xp_next.astype(np.float32)}
        )

        # Combine the reward and the value of the next state into the q-targets
        q_next = np.array([
            next_values[idx,next_actions[idx].argmax()]
            for idx in range(self.config['batch_size'])
        ])
        q_next *= (1.-xp_done)
        q_targets = (xp_rewards + self.config['gamma']*q_next)

        # Perform the update
        feed = {
            self.net.x_batch: xp_states.astype(np.float32),
            self.net.q_targets: q_targets[:,np.newaxis]*q_mask,
            self.net.q_mask: q_mask
        }
        _, summaries, step = self.session.run([self.net._train_op, self.summaries, self.net.global_step], feed_dict=feed)


        return summaries, step