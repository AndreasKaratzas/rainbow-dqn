
import gym
import numpy
import torch
import random
import datetime

from gym import RewardWrapper
from skimage import transform
from gym.spaces import Box

class ClipReward(RewardWrapper):
    """Clip reward to [min, max].

    Args:
        RewardWrapper ([type]): [description]
    """
    def __init__(self, env, min_r, max_r):
        super(ClipReward, self).__init__(env)
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        return numpy.clip(reward, self.min_r, self.max_r)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=numpy.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(numpy.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
