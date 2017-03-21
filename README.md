# Deep Q Networks in tensorflow

This is a side project to learn more about reinforcement learning. The goal is to have a relatively simple implementation of Deep Q Networks that can learn on (some) of the Atari Games. 
## Content
* **train_agent.py** contains the code to train and save the model. It will write summaries of the training reward per episode, the validation reward, the 
* **evaluate_agent.py** has code to load a learned model and let it run indefinitely. Currently only pong is supported. By default it will run the included model that is trained for 1000 episodes on Pong. It's performance is mixed. It plays ok, but loses most games. 
* **dqn.py** the deep q network implemented in tensorflow. The code supports standard DQN and DDQN. 
* **agent.py** class for interacting with the environement. 
* **config.py** contains the parameter settings for CartPole, AcroBot and Pong.
* **util.py** some basic helper functions and the replay memory implementation
## Tricks used
* Double DQN
* Loss clipping
* Andrej Karpathy's pre-processing

## Important details and observations
* On the atari games, the replay memory must use uint8 to limit memory usage
* It is normal that the code appears to learn not much for a couple of hours
* Even though the code learns well on the Pong evironment, over-estimation of the q-values still occurs on classic control tasks. (Have to check whether this is the case for cartpole with DDQN). This would indicate that Double DQN does not fully solve the overestimation problem, but just reduces or delays its effect. But I have to verify this.

## Things to be updated
* Verifying how to sample as quick as possible from the replay memory.
* Storing the replay memory directly on GPU?

## Learning on the Pong environment
Requires about 8 hours on a NVidia Titan X


## Limitations
The code is only tested on CartPole-v0, CartPole-v1, AcroBot-V0, Pong-v0


## Dependencies
* Tensorflow 1.0
* OpenAI gym
* Matplotlib
* Numpy

## References
1. DQN
2. DDQN
3. Tensorflow
4. OpenAI gym
