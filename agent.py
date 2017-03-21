import numpy as np
import tensorflow as tf
import gym
from util import Experience


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

        with tf.variable_scope('RL'):
            self.episode = tf.Variable(0.,name='episode')
            self.training_reward = tf.Variable(0.,name='training_reward')
            self.validation_reward = tf.Variable(0.,name='validation_reward')
            tf.summary.scalar(name='training_reward',tensor=self.training_reward)
            tf.summary.scalar(name='validation_reward',tensor=self.validation_reward)

        # Create tensorflow variables
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.summaries = tf.summary.merge_all()
        if log_dir is not None:
            self.train_writer = tf.summary.FileWriter(self.log_dir, self.session.graph)

        self.epsilon = 1.0
        self.steps = 0

    def assign_train_to_target(self):
        from pie.util.timing import tic,toc
        tic()
        if self.config['double_q']:
            vars = tf.trainable_variables()
            train_vars = [v for v in vars if v.name.startswith('Q_network/')]
            train_vars.sort(key=lambda x:x.name)
            target_vars = [v for v in vars if v.name.startswith('T_network/')]
            target_vars.sort(key=lambda x:x.name)
            self.session.run([v[0].assign(v[1]) for v in zip(target_vars,train_vars)])
        toc()

    def _update_training_reward(self,reward):
        self.session.run(self.training_reward.assign(reward))

    def _update_validation_reward(self,reward):
        self.session.run(self.validation_reward.assign(reward))

    def get_training_state(self):
        return self.replay_memory.get_last_state(self.config['state_time'])

    def sample_action(self,state,epsilon):
        """
        Sample an action

        :param state:
        :param epsilon:
        :return:
        """
        if np.random.rand() <= epsilon:
            return np.random.randint(0,self.config['actions'])
        else:
            return self.session.run(self.net.q, feed_dict={self.net.x: state[np.newaxis].astype(np.float32)})[0].argmax()

    def _update_state(self,old_state,new_frame):
        """

        :param old_state:
        :param new_frame:
        :return:
        """
        return np.concatenate([
            old_state[:, :, 1:],
            new_frame
        ], axis=2)

    def _reset_state(self):
        return np.concatenate([
                    np.zeros(self.config['state_shape']+[self.config['state_time']-1]),
                    self.config['frame'](self.env.reset())
                ],axis=2)

    def update_epsilon_and_steps(self):
        if self.steps > self.config['step_startrl']:
            self.epsilon = max(self.config['eps_minval'],self.epsilon*self.config['step_eps_mul']-self.config['step_eps_min'])
        self.steps += 1


    def train_episode(self):
        state = self._reset_state()
        for s_idx in range(state.shape[-1]-1):
            self.replay_memory.add(state[:,:,s_idx],np.random.randint(self.config['actions']),0,True)

        done = False
        total_reward = 0
        while not done:
            self.update_epsilon_and_steps()
            new_frame,reward,done = self.act(state,self.epsilon,True)
            state = self._update_state(state,new_frame)
            total_reward += reward

            if self.steps > self.config['step_startrl']:
                summaries,_ = self.train_batch()
                if self.steps % 1000 == 0:
                    self.train_writer.add_summary(summaries, global_step=self.steps)
                if self.steps % self.config['double_q_freq'] == 0.:
                    print "double q swap"
                    self.assign_train_to_target()

        return total_reward


    def validate_episode(self,epsilon,visualise=False):
        state = self._reset_state()
        done = False
        total_reward = 0.
        while not done:
            new_frame, reward, done = self.act(state=state, epsilon=epsilon, store=False)
            state = self._update_state(old_state=state, new_frame=new_frame)
            total_reward += reward
            if visualise:
                self.env.render()
        return total_reward

    def act(self,state,epsilon,store=False):
        """
        Perform an action in the environment

        :param epsilon:
        :param state:
        :param store:
        :return:
        """
        action = self.sample_action(state=state,epsilon=epsilon)
        raw_frame, reward, done, _ = self.env.step(action)

        # Clip rewards to -1,0,1
        reward = np.sign(reward)

        # Preprocess the output state
        new_frame = self.config['frame'](raw_frame)

        # If needed, store the last frame
        if store:
            self.replay_memory.add(state[:,:,-1],action,reward,done)
        return new_frame, reward, done


    def train_batch(self):
        """

        :return:
        """
        xp_states, xp_actions, xp_rewards, xp_done, xp_next = self.replay_memory.sample_experience(
            self.config['batch_size'],
            self.config['state_time']
        )

        # Create the mask for the training...
        q_mask = np.zeros((self.config['batch_size'], self.config['actions']))
        for idx, a in enumerate(xp_actions):
            q_mask[idx, a] = 1

        # Use the current network to select actions...
        next_actions, next_values = self.session.run(
            [self.net.q_batch,self.net.t_batch],
            feed_dict={self.net.x_batch: xp_next.astype(np.float32)}
        )

        q_next = np.array([
            next_values[idx,next_actions[idx].argmax()]
            for idx in range(self.config['batch_size'])
        ])
        q_next *= (1.-xp_done)

        q_targets = (xp_rewards + self.config['gamma']*q_next)

        feed = {
            self.net.x_batch: xp_states.astype(np.float32),
            self.net.q_targets: q_targets[:,np.newaxis]*q_mask,
            self.net.q_mask: q_mask
        }
        _, summaries, step = self.session.run([self.net._train_op, self.summaries, self.net.global_step], feed_dict=feed)
        return summaries, step