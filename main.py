"""Driver script.
"""

import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['DISPLAY'] = 'localhost:10.0'
# os.environ['SDL_VIDEODRIVER'] = 'dummy'

import sys
sys.path.append('./')

import coloredlogs
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from pathlib import Path

from envs.mario import create_mario_env

from src.args import get_options, info, load_args, store_args
from src.agent import Agent
from src.deterministic import set_deterministic, set_seed
from src.metrics import MetricLogger
from src.nvidia import cuda_check
from src.utils import create_data_dir, get_chkpt, experiment_data_plots
from src.engine import Engine
from src.wrappers import ResizeObservation, SkipFrame


if __name__ == "__main__":
    # Parse command line arguments
    args = get_options()
    
    # Print info
    if args.info:
        info()
    
    # check for CUDA compatible device
    cuda_check(True)

    # utilize deterministic algorithms for reproducibility and speed-up
    if args.use_deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
        set_deterministic(seed=args.seed)

    # Seed random number generators
    set_seed(args.seed)

    # Install logger
    coloredlogs.install(
        level='INFO', fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

    # Initialize output data directory
    export_data_dir = Path(args.data_dir)

    # Initialize output data filepaths
    model_save_dir, memory_save_dir, log_save_dir, datetime_tag = create_data_dir(
        export_data_dir)
    
    # Store arguments in a dictionary
    if args.store:
        if args.verbose:
            print(f"Exporting options to: {log_save_dir}")
        store_args(Path(log_save_dir) / Path(args.store + '.json'), args)
    
    # Sync arguments to file
    if args.sync:
        if args.verbose:
            print(f"Syncing options with: {log_save_dir}")
        load_args(Path(log_save_dir) / Path(args.sync + '.json'), args)

    # Get checkpoint
    model_checkpoint, mem_checkpoint = get_chkpt(args.resume)

    # Configure render mode
    if args.render:
        render_mode = "human" 
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "rgb_array"

    # Build environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode=render_mode)
    # Apply Wrappers to environment
    env = create_mario_env(env)
    # Reset environment
    env.reset(seed=args.seed)
    # Create Rainbow agent
    agent = Agent(
        env=env, batch_size=args.batch_size, target_sync=args.target_sync, gamma=args.gamma,
        num_of_steps_to_checkpoint_model=args.num_of_steps_to_checkpoint_model, beta=args.beta, 
        mem_capacity=args.mem_capacity, alpha=args.alpha, n_step=args.n_step, v_max=args.v_max,
        v_min=args.v_min, n_atoms=args.n_atoms, learning_rate=args.learning_rate, episodes=args.episodes,
        model_save_dir=model_save_dir, memory_save_dir=memory_save_dir, model_checkpoint=model_checkpoint,
        mem_checkpoint=mem_checkpoint, clip_grad_norm=args.clip_grad_norm, topk=args.top_k, verbose=args.verbose, 
        demo=args.test_mode, learning_starts=args.learning_starts, num_hiddens=args.num_hiddens, device=args.device,
        prior_eps=args.prior_eps, num_of_steps_to_checkpoint_memory=args.num_of_steps_to_checkpoint_memory,
        activation=args.activation, enable_base_model=args.use_base, )

    # Declare a logger instance to output experiment data
    logger = MetricLogger(log_save_dir, args.name, args.wandb, args.tensorboard, args.verbosity)

    # Log machine details
    logger.log_env_info()

    # Fire up engine
    Engine(
        env=env, agent=agent, logger=logger, en_train=args.train, en_eval=args.evaluate, 
        en_visual=args.render, episodes=args.episodes, test_cases=args.test_set_cardinality, 
        verbosity=args.verbosity, workload=args.workload,
    )

    if args.train:
        # Plot log data
        experiment_data_plots(export_data_dir, datetime_tag)
