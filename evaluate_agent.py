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
    config['state_memory']=1 # prevent allocating a lot

    param_dir = 'log/2017-03-20_17-01-26_Pong-v0_True'
    load_episode = 1000

    log_dir = None#get_log_dir('log', 'evaluation_'+config['game']+'_'+str(config['double_q']))
    with tf.device('/cpu:0'):
        agent = QAgent(config=config, log_dir=log_dir)
    saver = tf.train.Saver()
    saver.restore(agent.session,'%s/episode_%d.ckpt'%(param_dir,load_episode))
    agent.assign_train_to_target()

    while True:
        state = agent._reset_state()
        done = False
        total_reward = 0.

        # Prepare the visualisation
        plt.figure('q_network')
        plt.clf()
        plt.subplot(3,1,1)
        ax = plt.gca()
        bar = ax.bar(range(config['actions']),np.zeros(config['actions']))
        plt.xticks(.5+np.r_[:config['actions']],['None','None','Up','Down','Up','Down'])
        plt.ylabel('q[a]/max|q|')
        plt.subplot(3,1,2)
        ax = plt.gca()
        line_q, = ax.plot([], [], 'k', label='Q(s)', lw=2)
        #plt.subplot(3,1,3)
        #ax = plt.gca()
        line_r, = ax.plot([], [], 'xr', label='R(s)', lw=2,markersize=20)

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
            plt.subplot(3,1,1)
            plt.xlim(0,q.shape[1]-1)
            plt.ylim(-1,1)#[i*np.abs(q).max() for i in [-1.,1.]])
            plt.subplot(3,1,2)
            plt.xlim(0,len(qs))
            plt.ylim(-2,2)
            plt.subplot(3,1,3)
            plt.xlim(0,len(qs))
            plt.ylim(-2,2)

            agent.env.render()
            plt.draw()
            import time
            #time.sleep(0.01)