from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
plt.ion()

from plot_util import init_figure, update_figure

import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import pong_config, breakout_config

if __name__ == '__main__':
    config = pong_config
    config['state_memory']=1 # prevent allocating of a huge chunk of memory
    load_episode = 3650
    epsilon = 0.05 # The epsilon for the strategy

    # Build the graph on CPU to keep gpu for training....
    with tf.device('/cpu:0'):
        agent = QAgent(config=config, log_dir=None)

    # Restore the values....
    tf.train.Saver().restore(agent.session,'saves/%s/episode_%d.ckpt'%(config['game'],load_episode))


    while True:
        print("\n\n\n")
        # Initialise the episode
        state = agent.reset_to_zero_state()
        done = False
        total_reward = 0.
        steps = -1
        # Prepare the visualisation
        plots = init_figure(config['actions'])
        while not done:
            steps += 1
            q = agent.session.run(agent.net.q,feed_dict={agent.net.x:state[np.newaxis].astype(np.float32)})

            new_frame, reward, done,_ = agent.act(state=state, epsilon=epsilon, store=False)
            state = agent.update_state(old_state=state, new_frame=new_frame)
            total_reward += reward
            if reward != 0:
                print(reward)
            update_figure(plots, steps, q, reward, agent.env.render(mode='rgb_array'))
            plt.draw()
            plt.pause(0.001)
