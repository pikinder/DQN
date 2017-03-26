from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DQN(object):
    """

    """

    def __init__(self, batch_size, state_shape, num_actions, regul_param=0., optimiser=None,summaries=False):
        self.batch_size = batch_size
        self.num_actions = num_actions

        with tf.variable_scope('Q_input'):
            self.x_batch = tf.placeholder(tf.float32, [batch_size] + state_shape, name='x_batch')
            self.x = tf.placeholder(tf.float32, [1] + state_shape, name='x_single')

        with tf.name_scope('Q_param'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.regul_param = tf.Variable(regul_param, name='regul_param', trainable=False)

        with tf.variable_scope('Q_network') as scope:
            self.q = self._create_outputs(self.x)
            scope.reuse_variables()
            self.q_batch = self._create_outputs(self.x_batch)

        with tf.variable_scope('T_network') as scope:
            self.t_batch = self._create_outputs(self.x_batch)

        with tf.variable_scope('Q_losses'):
            self.q_targets = tf.placeholder(tf.float32, [self.batch_size, self.num_actions], name='q_targets')
            self.q_mask = tf.placeholder(tf.float32, [self.batch_size, self.num_actions], name='q_mask')
            self.loss = self._create_losses()

        with tf.variable_scope('Q_trainer'):
            self._train_op = self._create_train_ops(optimiser)

        if summaries:
            with tf.name_scope('Q_summary'):
                self._create_summaries()

        with tf.name_scope('Update_T'):
            self.assign_op = self._create_assign_op()

    def _create_assign_op(self):
        vars = tf.trainable_variables()
        train_vars = [v for v in vars if v.name.startswith('Q_network/')]
        train_vars.sort(key=lambda x:x.name)
        target_vars = [v for v in vars if v.name.startswith('T_network/')]
        target_vars.sort(key=lambda x:x.name)
        return  [v[0].assign(v[1]) for v in zip(target_vars,train_vars)]

    def _create_summaries(self):
        tf.summary.scalar(name='q_regul', tensor=self.regul_param)
        tf.summary.scalar(name='q_mse', tensor=self.loss)
        tf.summary.scalar(name='q_target_mean', tensor=tf.reduce_mean(self.q_targets * self.q_mask))

    def _create_outputs(self, x):
        """

        :param x:
        :return: the output of the tensorflow graph
        """
        raise NotImplementedError("Subclass responsability")

    def _atari_loss_hack(self,d):
        delta = 1.
        quadratic_part = tf.minimum(tf.abs(d), delta)
        linear_part = tf.abs(d) - quadratic_part
        return 0.5 * quadratic_part ** 2 + delta * linear_part

    def _squared_loss(self,d):
       return tf.square(d)

    def _create_losses(self):
        diff = tf.reduce_sum(tf.subtract(self.q_batch * self.q_mask,self.q_targets * self.q_mask),axis=1)
        loss = tf.reduce_mean(self._atari_loss_hack(diff))
        regulariser = self.regul_param * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + regulariser

    def _create_train_ops(self, optimiser):
        if optimiser is None:
            optimiser = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
        grads_and_vars = optimiser.compute_gradients(self.loss, var_list=[v for v in tf.trainable_variables() if v.name.startswith('Q_network/')])
        train_op = optimiser.apply_gradients(grads_and_vars=grads_and_vars, global_step=self.global_step)
        return train_op


class StateDQN(DQN):
    """
    DQN that acts on the state as a vector
    """

    def _create_outputs(self, x):
        x = tf.reshape(x, shape=[x.get_shape().as_list()[0], -1], name='flatten')
        x = tf.layers.dense(x, units=256, activation=tf.nn.relu, name='fc1')
        x = tf.layers.dense(x, units=256, activation=tf.nn.relu, name='fc2')
        return tf.layers.dense(x, units=self.num_actions, activation=None, name='q_hat')

class AtariDQN(DQN):
    """
    DQN that acts on a visual state
    """

    def _create_outputs(self, x):
        with tf.name_scope('shift_and_scale'):
            x = x/127.5-1.

        x = tf.layers.conv2d(x, filters=32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation=tf.nn.relu,
                             name='conv_1')
        x = tf.layers.conv2d(x, filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=tf.nn.relu,
                             name='conv_2')
        x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=tf.nn.relu,
                             name='conv_3')
        x = tf.reshape(x, shape=[x.get_shape().as_list()[0], -1], name='flatten')
        x = tf.layers.dense(x, units=512, activation=tf.nn.relu, name='fc1')
        return tf.layers.dense(x, units=self.num_actions, activation=None, name='q_hat')

    def _create_summaries(self):
        super(AtariDQN, self)._create_summaries()
        tf.summary.image(name='x_batch', tensor=self.x_batch[:, :, :, 1:], max_outputs=8)
