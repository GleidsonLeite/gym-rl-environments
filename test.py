from random import random
from stable_baselines3.common.env_checker import check_env
import numpy as np
from src.modules.LNA.entities import LNA

env = LNA()

observation = env.reset()
sample_action = env.action_space.sample()
new_observation, reward, done, _ = env.step(sample_action / 10)
print(new_observation)
print(reward)
print(observation)
# print(test)
# print(np.shape(env_sample))
# print(np.shape(test))
# print(np.shape(env_sample) == np.shape(test))

# print(env.observation_space.contains(test))
