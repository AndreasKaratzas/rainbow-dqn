
import numpy as np
import cv2
import collections
import gymnasium as gym

from gymnasium import (
    RewardWrapper, ObservationWrapper)
from gymnasium.wrappers import (
    FrameStack, GrayScaleObservation, 
    ResizeObservation, AtariPreprocessing)


class ClipReward(RewardWrapper):
    """Clip reward to [min, max].

    Parameters
    ----------
    env : gym.Env
        Environment to wrap.
    min_r : float
        Minimum reward.
    max_r : float
        Maximum reward.
    """
    def __init__(self, env, min_r, max_r):
        super(ClipReward, self).__init__(env)
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        """Clip reward to [min, max].

        Parameters
        ----------
        reward : float
            Reward to clip.

        Returns
        -------
        float
            Clipped reward.
        """
        return np.clip(reward, self.min_r, self.max_r)


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
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info
    
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _, info = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _, info = self.env.step(2)
        if done:
            self.env.reset()
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, truncated, info
    
    def reset(self):
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, info

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    
    def observation(self, obs):
        return ProcessFrame84.process(obs)
    
    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160,  3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            print(f"frame size is {frame.size}")
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
            x_t = resized_screen[18:102, :]
            x_t = np.reshape(x_t, [84, 84, 1])
            return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(
            n_steps, axis=0), old_space.high.repeat(n_steps, axis=0), dtype=dtype)
    
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset()
        return self.observation(obs), info
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(
            old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env, val=255.0):
        super(ScaledFloatFrame, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=old_shape, dtype=np.float32)
        self.val = val

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / self.val


def wrap_env(env, reward_clipping=None, frame_stack=None, val=None):
    """Wrap environment with reward clipping and frame stacking.

    Parameters
    ----------
    env : gym.Env
        Environment to wrap.
    reward_clipping : tuple of float, optional
        Tuple of (min, max) reward clipping values.
    frame_stack : int, optional
        Number of frames to stack.
    val : int, optional
        Value to normalize the observation space by.

    Returns
    -------
    gym.Env
        Wrapped environment.
    """
    if reward_clipping is not None:
        env = ClipReward(env, *reward_clipping)
    if frame_stack is not None:
        env = FrameStack(env, frame_stack)
    if val is not None:
        env = ScaledFloatFrame(env, val)
    return env


def demo_wrap_env_bak_mine(env, reward_clipping=None, frame_stack=None):
    """Demo wrap_env."""
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(64, 64))
    env = wrap_env(env, reward_clipping=reward_clipping, frame_stack=frame_stack)
    return env


def demo_wrap_env_bak(env, reward_clipping=None, frame_stack=None):
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env) 
    env = BufferWrapper(env, frame_stack)
    return ScaledFloatFrame(env)


def demo_wrap_env(env, reward_clipping=None, frame_stack=None):
    env = AtariPreprocessing(env, frame_skip=1, noop_max=30, terminal_on_life_loss=True)
    env = FrameStack(env, frame_stack)
    return env
    