# Deep Q Networks in tensorflow

This is a side project to learn more about reinforcement learning. 
The goal is to have a relatively simple implementation of Deep Q Networks [1,2] that can learn on (some) of the Atari Games. 
_It is not an exact reproduction of the original paper._


## Content
* **train_agent.py** contains the code to train and save the model. It will write summaries of the training reward per episode, the validation reward, the mse, the regularisation parameter, the mean target q value.
* **evaluate_agent.py** has code to load a learned model and let it run indefinitely. Currently only pong is supported. By default it will run the included model that is trained for 1550 episodes on Pong. It's performance is mixed. It plays ok, but loses most games. The script shows the following visualisation of game, q-function and value history+reward.
![alt text](readme/evaluation_output.png?raw=true "evaluation visualisation")

* **dqn.py** the deep q network implemented in tensorflow. The code supports standard DQN [1] and Double DQN [3]. 
* **agent.py** class for interacting with the environement. 
* **replay.py** replay memory implementation
* **config.py** contains the parameter settings for CartPole, AcroBot and Pong.
* **util.py** some basic helper functions
* **saves/** Checkpoints networks that work reliably on the pong environment

## Implementation details
* The architecture from DeepMind's atari nature publication [2].
* Support for standard DQN (without target network) [1] and Double DQN [3].
* Loss clipping from DeepMind's nature paper[2]. For the implementation I looked at [6]. 
* Andrej Karpathy's pre-processing was used here [7]. This works only on breakout (I think) and pong (for sure). 
* On the atari games, the replay memory must use uint8 to limit memory usage.

# Notes
* It is normal that the code appears to learn not much for a couple of hours. Requires about 8 hours on a NVidia Titan X to do something sensible on Pong. 
* Even though the code learns well on the Pong evironment, over-estimation of the q-values still occurs on classic control tasks. (Have to check whether this is the case for cartpole with DDQN). This would indicate that Double DQN does not fully solve the overestimation problem, but just reduces or delays its effect. But I have to verify this.

## Things to be updated
* Verifying how to sample as quick as possible from the replay memory.
* Storing the replay memory directly on GPU?

## Limitations
The code is only tested on CartPole-v0, CartPole-v1, AcroBot-V0, Pong-v0


## Dependencies
* Tensorflow 1.0
* OpenAI gym
* Matplotlib
* Numpy

## References
1. [Mnih et al. Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
2. [Mnih et al. Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
3. [van Hasselt et al. Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
4. [Abadi et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://research.google.com/pubs/pub45166.html)
5. [OpenAI gym](https://gym.openai.com)
6. [Nathan Sprague's theano DQN implementation](https://github.com/spragunr/deep_q_rl)
7. [Karpathy's blog post on RL with OpenAI gym](http://karpathy.github.io/2016/05/31/rl/)

