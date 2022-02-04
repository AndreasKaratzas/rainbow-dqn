"""Driver script.
"""

import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from pathlib import Path

from envs.mario import create_mario_env

from src.core import *
from src.test import *
from src.train import *
from src.functions import *
from src.agent import Daedalous
from src.metrics import MetricLogger
from src.wrappers import ResizeObservation, SkipFrame


if __name__ == "__main__":
    # Seed random number generators
    seed_generators(SEED)
    # Initialize output data directory
    export_data_dir = Path("data")
    # Initialize output data filepaths
    model_save_dir, memory_save_dir, log_save_dir, datetime_tag = create_data_dir(
        export_data_dir)
    # Get latest checkpoint data
    model_checkpoint, mem_checkpoint = get_last_chkpt(
        export_data_dir, "2022-02-03T00-38-29", True, True, False)
    # Build environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    # Apply Wrappers to environment
    env = create_mario_env(env)
    # Reset environment
    env.reset()
    # Create Rainbow agent
    daedalous = Daedalous(env, model_save_dir, memory_save_dir,
                          model_checkpoint, mem_checkpoint)
    # Declare a logger instance to output experiment data
    logger = MetricLogger(log_save_dir)
    # Fit agent
    train(env, daedalous, logger)
    # Test agent
    test(env, daedalous)
    # Plot log data
    experiment_data_plots(export_data_dir, "2022-02-03T00-38-29")
