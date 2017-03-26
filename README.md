# Deep Q Networks in tensorflow

This is a side project to learn more about reinforcement learning. 
The goal is to have a relatively simple implementation of Deep Q Networks [1,2] that can learn on (some) of the Atari Games. 
_It is not an exact reproduction of the original paper._

## Notes
* The neural net architecture from DeepMind's atari nature publication [2] is used.

* The code supports standard DQN (without target network) [1] and Double DQN [3].
* Loss clipping from DeepMind's nature paper[2] is used. For the implementation I looked at [6]. 
* For pre-processing, I crop according to Andrej Karpathy's pre-processing [7]. Convert to RGB and rescale to 84 by 84.
* On the atari games, the replay memory must use uint8 to limit memory usage.
* I use _PongDeterministic-v3_ and _BreakOutDeterministic_v3_. This uses the same deterministic frame skipping as in the deepmind publications (4 frames). It also disables random actions on the skipped frames. This makes learning much faster compared to the _Pong-v0_ and _BreakOut-v0_ environments (rough estimate: 4 times faster on pong). 
* Using a Nvidia Titan X on Pong, the model performs sensible actions after about 2-3 hours. At this point it is not able to beat the AI. To beat the AI a bit more than 1.3*10**6 frames are needed. To get to this point it took 10 hours of training time, which is short for these games. I did not train longer because of my limited number of GPUs

## Content
* **train_agent.py** contains the code to train and save the model. It will write summaries of the training reward per episode, the validation reward, the mse, the regularisation parameter, the mean target q value.
* **evaluate_agent.py** has code to load a trained model and let it run indefinitely. Currently only Atari Games are supported. By default it will run the included model that is trained for 750 episodes on Pong. It's performance is ok, it can win games but it is not unbeatable. The script shows the following visualisation of game, q-function and value history+reward.
![alt text](readme/evaluation_output.png?raw=true "evaluation visualisation")

* **dqn.py** the deep q network implemented in tensorflow. The code supports standard DQN [1] and Double DQN [3]. 
* **agent.py** class for interacting with the environement. 
* **replay.py** replay memory implementation
* **config.py** contains the parameter settings for CartPole, Pong and BreaKout.
* **util.py** some basic helper functions
* **saves/** Checkpoints of networks that work reliably
* **log/** directory where the tensorboard summaries and the checkpoints are written to.


## Things to be updated
* Verifying how to sample as quick as possible from the replay memory.
* Storing the replay memory directly on GPU?
* Making sure the GPU is better utilised. 
* Ensure that the pre-processing works on all games

## Dependencies
* Tensorflow 1.0
* OpenAI gym
* Matplotlib
* Numpy
* skimage for grayscale and resizing

## References
1. [Mnih et al. Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
2. [Mnih et al. Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
3. [van Hasselt et al. Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
4. [Abadi et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://research.google.com/pubs/pub45166.html)
5. [OpenAI gym](https://gym.openai.com)
6. [Nathan Sprague's theano DQN implementation](https://github.com/spragunr/deep_q_rl)
7. [Karpathy's blog post on RL with OpenAI gym](http://karpathy.github.io/2016/05/31/rl/)

