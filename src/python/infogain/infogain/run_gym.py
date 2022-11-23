import gym
import json
import datetime as dt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from OcclusionGym import OcclusionEnv

# Parallel environments
env = make_vec_env(OcclusionEnv, n_envs=4, env_kwargs={ 'num_actors': 10 })

model = PPO("MlpPolicy", env, verbose=1)
try:
    model.load( 'ppo_occlusions.model' )
except IOError:
    pass 
model.learn(total_timesteps=25000)

model.save( 'ppo_occlusions.model')

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()
