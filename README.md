# Deep Q Networks in tensorflow

This is a side project to learn more about reinforcement learning. 
The goal is to have a relatively simple implementation of Deep Q Networks [1,2] that can learn on (some) of the Atari Games. 
_It is not an exact reproduction of the original paper._

## Notes
* The architecture from DeepMind's nature publication [2] is used.
* Standard DQN (without target network) [1] and Double DQN [3] is implemented.
* Loss clipping from DeepMind's nature paper [2] is used. ( The implementation mimics [6].)
* Pre-processing is done by
    1. RGB to grayscale conversion
    2. Rescaling to 84 by 84 (this does not preserve the aspect ratio).
* On the atari games, the replay memory uses uint8 to reduce memory usage.
* The atari games are accessed through OpenAI Gym [5] but not using the default environments.
    1.  _PongDeterministic-v3_ and _BreakOutDeterministic_v3_ are used.
            This used deterministic frame skipping and action repeating similar to [2].
            Consequently it learns about 4 times faster compared to the less deterministic _Pong-v0_  environment.
    2. The loss of a life results in a terminal state. This was used by Mnih at al. in [2].
        
## Content
* **train_agent.py** contains the code to train and save the model. It will write summaries of the training reward per episode, the validation reward, the mse, the regularisation parameter, the mean target q value.
* **evaluate_agent.py** has code to load a trained model and let it run indefinitely. 
    The script shows the following visualisation of game, q-function and value history+reward.
![alt text](readme/evaluation_output.png?raw=true "evaluation visualisation")
* **dqn.py** the deep q network implemented in tensorflow. The code supports standard DQN [1] and Double DQN [3]. 
* **agent.py** class for interacting with the environment. 
* **replay.py** replay memory implementation
* **config.py** contains the parameter settings for CartPole, Pong and Breakout.
* **util.py** some basic helper functions
* **saves/** Checkpoints of networks that work reliably
* **log/** directory where the tensorboard summaries and the checkpoints are written to.


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
