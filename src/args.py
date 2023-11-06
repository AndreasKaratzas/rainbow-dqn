
import argparse
import json

from copy import deepcopy
from common.terminal import colorstr


def info():
    print(f"\n\n"
          f"\t\t      The {colorstr(['red', 'bold'], list(['Rainbow DQN']))} software operates as a RL framework \n"
          f"\t\t for efficient discrete action space exploration. \n"
          f"\n")
          

def get_options():
    """Example run:
    
    >>> python main.py --device 'cuda:0' --mem-capacity 100000 --batch-size 128 --target-sync 1000 --learning-starts 100000 --num-of-steps-to-checkpoint-model 1000 --num-of-steps-to-checkpoint-memory 10000 --episodes 10000 --gamma 0.99 --alpha 0.4 --beta 0.6 --prior-eps 1e-6 --v-min -21.0 --v-max 20.0 --n-atoms 51 --n-step 3 --verbosity 10 --learning-rate 2.5e-4 --info --use-deterministic-algorithms --clip-grad-norm 40.0 --top-k 10 --num-hiddens 512 --wandb --tensorboard --name "FairBoost" --store "options" --verbose --activation 'gelu' --seed 33 --workload 0 12 10 1 --train
    """
    parser = argparse.ArgumentParser(
        description=f'Parser for the Rainbow DQN.\n'
                    f'For information on the Rainbow DQN software, pass `--info`')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--mem-capacity', type=int,
                        default=int(1e5), help='memory capacity')
    parser.add_argument('--batch-size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--target-sync', type=int,
                        default=int(1e3), help='target sync')
    parser.add_argument('--learning-starts', type=int,
                        default=int(1e5), help='learning starts')
    parser.add_argument('--num-of-steps-to-checkpoint-model', type=int,
                        default=int(1e3), help='num of steps to checkpoint model')
    parser.add_argument('--num-of-steps-to-checkpoint-memory', type=int,
                        default=int(1e6), help='num of steps to checkpoint experience replay')
    parser.add_argument('--episodes', type=int,
                        default=int(1e7), help='episodes')
    parser.add_argument("--workload", type=int, nargs='+', 
                        help="The neural network workload to be simulated.")
    parser.add_argument('--activation', type=str,
                        default='relu', help='activation')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--alpha', type=float, default=0.4, help='alpha')
    parser.add_argument('--beta', type=float, default=0.6, help='beta')
    parser.add_argument('--prior-eps', type=float,
                        default=1e-6, help='prior eps')
    parser.add_argument('--use-base', action='store_true', help='use base model')
    parser.add_argument('--v-min', type=float, default=-21.0, help='v min')
    parser.add_argument('--v-max', type=float, default=20.0, help='v max')
    parser.add_argument('--n-atoms', type=int, default=51, help='n atoms')
    parser.add_argument('--n-step', type=int, default=3, help='n step')
    parser.add_argument('--verbosity', type=int, default=10, help='verbosity')
    parser.add_argument('--learning-rate', type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument('--test-set-cardinality', type=int,
                        default=10, help='test set cardinality')
    parser.add_argument("--info", action="store_true",
                        help="print abstract software info.")
    parser.add_argument("--seed", default=33, type=int,
                        help="seed used to reproduce an experiment")
    parser.add_argument("--verbose", action="store_true",
                        help="print the model architecture and other information.")
    parser.add_argument("--name", default="FairBoost", type=str,
                        help="alias for the running experiment")
    parser.add_argument("--use-deterministic-algorithms", action="store_true",
                        help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--clip-grad-norm", default=40.0, type=float,
                        help="the maximum gradient norm (default 40.0)")
    parser.add_argument("--resume", default="", type=str,
                        help="path of checkpoint")
    parser.add_argument("--train", action="store_true",
                        help="train a model model")
    parser.add_argument("--evaluate", action="store_true",
                        help="validate a model model")
    parser.add_argument("--no-render", action="store_true",
                        help="do not render the environment")
    parser.add_argument("--data-dir", default="data",
                        type=str, help="data directory")
    parser.add_argument("--top-k", default=5, type=int,
                        help="top-k models to store in backup directory")
    parser.add_argument("--sync", default=None, type=str,
                        help="load arguments from a file")
    parser.add_argument("--store", default=None, type=str,
                        help="store arguments in a file")
    parser.add_argument("--wandb", action="store_true",
                        help="use wandb for logging")
    parser.add_argument("--tensorboard", action="store_true",
                        help="use tensorboard for logging")
    parser.add_argument("--obs-stack", default=4, type=int,
                        help="number of observations to stack")
    parser.add_argument("--norm-value", default=0.0, type=float,
                        help="value to normalize the observations")
    parser.add_argument("--demo", action="store_true", help="Set driver to demo mode (for debugging purposes)")
    parser.add_argument("--test-mode", action="store_true", help="Set agent to test mode")
    parser.add_argument("--render", action="store_true", help="render mode")
    parser.add_argument("--num-hiddens", default=512,
                        type=int, help="number of hidden units")
    return parser.parse_args()


def store_args(filename, args):
    """Store the arguments in a file.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Parser for the Simulator library.
    filename : str
        Path to the file where the arguments are stored.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the arguments.
    """
    with open(filename, 'w+') as f:
        json.dump(args.__dict__, f, indent=2)
    
    return args.__dict__


def load_args(filename, args):
    """Load the arguments from a file.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Parser for the Simulator library.
    filename : str
        Path to the file where the arguments are stored.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the Simulator library.
    """
    backup = deepcopy(args)
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)
    
    keys = backup.__dict__.keys()
    for key in keys:
        if key not in args.__dict__:
            args.__dict__[key] = backup.__dict__[key]

    return args
