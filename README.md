# Deep Q Networks in tensorflow

This is a side project to learn more about reinforcement learning. The goal is to have a relatively simple implementation of Deep Q Networks that can learn on (some) of the Atari Games. 

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
