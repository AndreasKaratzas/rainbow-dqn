
import time
import numpy
import datetime

from src.core import *


class MetricLogger():
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'ReturnMin':>15}{'ReturnMax':>15}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = numpy.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
        
        self.ep_avg_losses.append(ep_avg_loss)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, step):
        mean_ep_reward = numpy.round(
            numpy.mean(self.ep_rewards[-VERBOSITY:]), 3)
        mean_ep_length = numpy.round(
            numpy.mean(self.ep_lengths[-VERBOSITY:]), 3)
        mean_ep_loss = numpy.round(numpy.mean(
            self.ep_avg_losses[-VERBOSITY:]), 3)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = numpy.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode:>7} - "
            f"Step {step:>10} - "
            f"Min Return {numpy.round(numpy.min(self.ep_rewards[-VERBOSITY:]), 3):>8.3f} - "
            f"Max Return {numpy.round(numpy.max(self.ep_rewards[-VERBOSITY:]), 3):>8.3f} - "
            f"Mean Reward {mean_ep_reward:>10.3f} - "
            f"Mean Length {mean_ep_length:>10.3f} - "
            f"Mean Loss {mean_ep_loss:>8.3f} - "
            f"Time Delta {time_since_last_record:>9.3f} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}"
                f"{numpy.round(numpy.min(self.ep_rewards[-VERBOSITY:]), 3):15.3f}{numpy.round(numpy.max(self.ep_rewards[-VERBOSITY:]), 3):15.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
