from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from gym.envs.atari.atari_env import ACTION_MEANING
def disable_ticks():
    plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off') # labels along the bottom edge are off
    plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off') # labels along the bottom edge are off
    plt.axis('off')


def init_figure(actions):
    plt.figure('q_network',figsize=(15,10))
    plt.clf()

    plots = {}

    plt.subplot(1,2,1)
    ax = plt.gca()
    plots['screen']=(plt.imshow(np.zeros((210,160,3)),interpolation='nearest'),ax)
    disable_ticks()
    plt.subplot(2,2,2)
    plt.title('Q function')
    ax = plt.gca()
    plots['Qs'] = (ax.bar(range(actions),np.zeros(actions)),ax)
    plt.xticks(.5+np.r_[:actions],[ACTION_MEANING[a] for a in range(actions)])
    plt.ylabel('Q(s,a)')
    plt.ylim(-2,2)
    plt.subplot(2,2,4)
    ax = plt.gca()
    line_q, = ax.plot([], [], 'k', label='max(Q(s,.))', lw=2)
    plots['Qhist']=(line_q,ax)
    line_r, =  ax.plot([], [], 'or', label='R(s,a)', lw=10)
    plots['Rhist'] = (line_r,ax)


    # Create a legend to the left...
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, 0.8*box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(.5,1.3))
    plt.xlabel('frame')
    plt.ylabel('Reward/max Q')

    return plots


def extend_line(line,pair):
    x,y= line.get_data()
    x = np.hstack([x,pair[0]])
    y = np.hstack([y,pair[1]])
    line.set_data(x,y)


def update_figure(plots,step,q,r,img,max_len=0):
    for rect, yi in zip(plots['Qs'][0], q[0]):
        rect.set_height(yi)
    extend_line(plots['Qhist'][0],(step,q.max()))
    if r!=0:
        extend_line(plots['Rhist'][0],(step,r))
    if max_len:
        plots['Qhist'][1].set_xlim([step-max_len,step])
    else:
        plots['Qhist'][1].set_xlim([0,step])
    plots['Qhist'][1].set_ylim([i*max(1.1,np.max(np.abs(plots['Qhist'][0].get_data()[1]))) for i in [-1.,1.]])
    plots['screen'][0].set_data(img)
    plt.draw()