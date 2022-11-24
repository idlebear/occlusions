import gym
import json
import datetime as dt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from OcclusionGym import OcclusionEnv

# Parallel environments
env = make_vec_env(OcclusionEnv, n_envs=4, env_kwargs={'num_actors': 10})


model = PPO("MlpPolicy", env, verbose=1)
try:
    model.load('ppo_occlusions.model')
    print('Previous model loaded')
except IOError:
    print('No model to load -- starting fresh')

# for i in range(10):
#     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./occlusion_log")
#     try:
#         model.load('ppo_occlusions.model')
#         print('Previous model loaded')
#     except IOError:
#         print('No model to load -- starting fresh')

#     model.learn(total_timesteps=50000)

#     model.save('ppo_occlusions.model')

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
