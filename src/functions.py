
import os
import math
import torch
import numpy
import pandas
import random
import datetime

import matplotlib.pyplot as plt

from pathlib import Path
from sys import platform


def list_all_subdirectories(export_data_path=Path("data")):
    return [f.path for f in os.scandir(export_data_path) if f.is_dir()]

def get_dir_size(export_data_path=Path("data")):
    return sum(f.stat().st_size for f in export_data_path.glob('**/*') if f.is_file())

def get_chkpt_datetime(export_data_path):
    l = list_all_subdirectories(export_data_path)
    s = numpy.zeros((len(l), 1), dtype=numpy.float)

    for idx, e in enumerate(l):
        s[idx] = get_dir_size(Path(e))
    
    delimiter = ''
    if platform == "win32":
        delimiter = '\\'
    elif platform == "linux" or platform == "linux2":
        delimiter = '/'
    else:
        raise ValueError('Found OS environment which has not been tested with the code.')

    l_datetime = [element.split(delimiter)[-1] for element in l]
    arg_sorted = numpy.argsort(-s, axis=0)
    normalized_diff = numpy.abs((s[arg_sorted[0]] - s[arg_sorted[1]])).item() / s[arg_sorted[0]].item()
    list_sorted = sorted(l_datetime, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H-%M-%S'))
    checkpoint_idx = arg_sorted[0].item()

    if normalized_diff < 1e-2:
        if list_sorted[arg_sorted[1].item()] > list_sorted[arg_sorted[0].item()]:
            checkpoint_idx = arg_sorted[1].item()

    return list_sorted[checkpoint_idx]

def seed_generators(seed):
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        print("cudnn status\t[enabled]")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def stats_printer(d, indent=0):
    numpy.set_printoptions(precision=4, suppress=True, formatter={'float': '{: 0.3f}'.format})
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            stats_printer(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def create_data_dir(export_data_path):
    datetime_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_save_dir = export_data_path / datetime_tag / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    memory_save_dir = export_data_path / datetime_tag / "memory"
    memory_save_dir.mkdir(parents=True, exist_ok=True)
    log_save_dir = export_data_path / datetime_tag / "log"
    log_save_dir.mkdir(parents=True, exist_ok=True)
    return model_save_dir, memory_save_dir, log_save_dir, datetime_tag

def get_last_chkpt(data_path, fixed_datetime, overwrite=False, load_model_chkpt=False, load_memory_chkpt=False):
    chkpt_datetime_dir = fixed_datetime if overwrite else get_chkpt_datetime(data_path)
    model_checkpoint = Path(data_path / chkpt_datetime_dir / "model")
    mem_checkpoint = Path(data_path / chkpt_datetime_dir / "memory")

    if not model_checkpoint.is_dir() or not load_model_chkpt:
        model_checkpoint = None
    if not mem_checkpoint.is_dir() or not load_memory_chkpt:
        mem_checkpoint = None
        
    return model_checkpoint, mem_checkpoint

def experiment_data_plots(export_data_dir, datetime_tag):
    """Plots training loss progress and mean accumulated episode reward.
    """
    filename = export_data_dir / datetime_tag / "log" / "log"
    df = pandas.read_csv(filename, delim_whitespace=True)
    plt.subplot(2, 1, 1)
    plt.title('Training stats')
    plt.plot(df.Step, df.MeanLoss)
    plt.ylabel('Mean Loss')
    plt.subplot(2, 1, 2)
    plt.plot(df.Step, df.MeanReward)
    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.tight_layout()
    plt.show()

def test_stats(towards_target, mission_status):
    """Gives important percentages regarding the model's test process.
    """

    labels_1 = ["Converged", "Diverged"]
    labels_2 = ["Success", "Failed"]

    y_1 = numpy.array([sum(towards_target), len(towards_target) - sum(towards_target)], dtype=numpy.int)
    y_2 = numpy.array([sum(mission_status), len(mission_status) - sum(mission_status)], dtype=numpy.int)

    fig, (ax_1, ax_2) = plt.subplots(2, 1)
    fig.suptitle('Test stats')

    ax_1.pie(y_1, labels=labels_1, autopct='%1.1f%%')
    ax_1.title.set_text('Action correctness w.r.t Target')

    ax_2.pie(y_2, labels=labels_2, autopct='%1.1f%%')
    ax_2.title.set_text('Mission success rate')
    
    plt.tight_layout()
    plt.show()
