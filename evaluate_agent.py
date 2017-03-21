from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import pong_config


if __name__ == '__main__':
    config = pong_config
    config['state_memory']=1 # prevent allocating of a huge chunk of memory
    epsilon = 0.00 # The epsilon for the strategy
    params = 'saves/PONG_1000.ckpt'
    agent = QAgent(config=config, log_dir=None)
    tf.train.Saver().restore(agent.session,params)

    # Play the game
    while True:
        # Initielise the episode
        state = agent._reset_state()
        done = False
        total_reward = 0.

        # Prepare the visualisation
        plt.figure('q_network')
        plt.clf()
        plt.subplot(2,1,1)
        ax = plt.gca()
        bar = ax.bar(range(config['actions']),np.zeros(config['actions']))
        plt.xticks(.5+np.r_[:config['actions']],['None','None','Up','Down','Up','Down'])
        plt.ylabel('q[a]/max|q|')
        plt.subplot(2,1,2)
        ax = plt.gca()
        line_q, = ax.plot([], [], 'k', label='max(Q(s,.))', lw=2)
        line_r, = ax.plot([], [], 'or', label='R(s,a)', lw=10)
        # Shrink current axis by 20%
        plt.legend()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('frames')
        plt.ylabel('reward')
        qs = []
        rx = []
        rs = []

        while not done:
            q = agent.session.run(agent.net.q,feed_dict={agent.net.x:state[np.newaxis].astype(np.float32)})

            new_frame, reward, done = agent.act(state=state, epsilon=0.01, store=False)
            state = agent._update_state(old_state=state, new_frame=new_frame)
            total_reward += reward

            qs.append(q.max())
            if reward!=0:
                rx.append(len(qs)-1)
                rs.append(reward)

            # Visualise...
            for rect, yi in zip(bar, q[0]):
                rect.set_height(yi/np.abs(q).max())
            line_q.set_data(range(len(qs)),qs)
            line_r.set_data(rx,rs)
            plt.subplot(2,1,1)
            plt.xlim(0,q.shape[1]-1)
            plt.ylim(-1,1)#[i*np.abs(q).max() for i in [-1.,1.]])
            plt.subplot(2,1,2)
            plt.xlim(0,len(qs))
            plt.ylim(-2,2)

            agent.env.render()
            plt.draw()
            import time
            #time.sleep(0.01)