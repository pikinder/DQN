from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from agent import QAgent
from configs import pong_config, breakout_config
from util import get_log_dir

if __name__ == '__main__':
    config = breakout_config
    log_dir = get_log_dir('log', config['game']+'_'+str(config['double_q']))
    agent = QAgent(config=config, log_dir=log_dir)
    saver = tf.train.Saver()
    for episode in range(config['episodes']):
        print('\n\nepisode: %d, step: %d, eps: %.4f\n\n---------------------' % (episode, agent.steps, agent.epsilon))
        # Store the rewards...
        agent._update_training_reward(agent.train_episode())

        if episode % config['episodes_validate']==0:
            print('Validate....\n==============')
            scores = [agent.validate_episode(epsilon=0.05) for i in range(config['episodes_validate_runs'])]
            agent._update_validation_reward(np.mean(scores))
            print(scores)
        # Store every validation interval
        if episode % config['episodes_save_interval']==0:
            saver.save(agent.session,'%s/episode_%d.ckpt'%(log_dir,episode))