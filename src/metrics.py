
import time
import datetime
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
import wandb

from src.info import collect_env_details


class MetricLogger:
    def __init__(
        self, 
        save_dir: str = '.', 
        name: str = None, 
        en_wandb: bool = False,
        en_tensorboard: bool = False,
        verbosity: int = 100
    ):
        self.save_log = save_dir / "agent.log"
        self.save_log.write_text(
            f"{'Episode':>8}{'Step':>8}{'ReturnMin':>15}{'ReturnMax':>15}{'MeanReward':>15}"
            f"{'MeanLength':>15}{'MeanLoss':>15}"
            f"{'TimeDelta':>15}{'Time':>20}\n"
        )

        self.last_ep = 0

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []

        self.init_episode()

        self.fetch_time = time.time()
        self.record_time = time.time()
        self.verbosity = verbosity
        
        if en_wandb:
            wandb.init(project=name)

        if en_tensorboard:
            # initialize tensorboard instance
            """To start a tensorboard instance, run the following command:
            
                >>> tensorboard --logdir=./experiment/ --host localhost --port 8888
            """
            self.tensorboard_writer = SummaryWriter(
                log_dir=save_dir)
        else:
            self.tensorboard_writer = None

        self.env_info_log = save_dir / "env_info.txt"

    def log_env_info(self):
        env_details = collect_env_details()
        with self.env_info_log.open("w") as f:
            f.write(env_details)

        # Log environment details to WandB as text
        if wandb.run:
            wandb.log({"Environment Info": wandb.Html(env_details.replace("\n", "<br>"), inject=False)})

    def log_step(self, reward, loss):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
        
        self.ep_avg_losses.append(ep_avg_loss)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_loss_length = 0
    
    def fetch(self, episode, step):

        if self.last_ep == episode:
            msg = (
                f"Episode {episode:>7} - "
                f"Step {step:>10} - "
                f"Min Return {np.round(self.curr_ep_reward, 3):>8.3f} - "
                f"Max Return {np.round(self.curr_ep_reward, 3):>8.3f} - "
                f"Mean Reward {np.round(self.curr_ep_reward, 3):>10.3f} - "
                f"Mean Length {np.round(self.curr_ep_length, 3):>10.3f} - "
                f"Mean Loss {np.round(self.curr_ep_loss, 3):>8.3f} - "
                f"Time Delta {np.round(time.time() - self.fetch_time, 3):>9.3f}"
            )

        else:
            mean_ep_reward = np.round(
                np.mean(self.ep_rewards[-self.verbosity:]), 3)
            mean_ep_length = np.round(
                np.mean(self.ep_lengths[-self.verbosity:]), 3)
            mean_ep_loss = np.round(np.mean(
                self.ep_avg_losses[-self.verbosity:]), 3)
            
            last_fetch_time = self.fetch_time
            self.fetch_time = time.time()
            time_since_last_fetch = np.round(
                self.fetch_time - last_fetch_time, 3)
            
            msg = (
                f"Episode {episode:>7} - "
                f"Step {step:>10} - "
                f"Min Return {np.round(np.min(self.ep_rewards[-self.verbosity:]), 3):>8.3f} - "
                f"Max Return {np.round(np.max(self.ep_rewards[-self.verbosity:]), 3):>8.3f} - "
                f"Mean Reward {mean_ep_reward:>10.3f} - "
                f"Mean Length {mean_ep_length:>10.3f} - "
                f"Mean Loss {mean_ep_loss:>8.3f} - "
                f"Time Delta {time_since_last_fetch:>9.3f}"
            )

            self.last_ep = episode

        return msg

    def record(self, episode, step, verbose=False):
        mean_ep_reward = np.round(
            np.mean(self.ep_rewards[-self.verbosity:]), 3)
        mean_ep_length = np.round(
            np.mean(self.ep_lengths[-self.verbosity:]), 3)
        mean_ep_loss = np.round(np.mean(
            self.ep_avg_losses[-self.verbosity:]), 3)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(
            self.record_time - last_record_time, 3)

        # Log metrics to WandB
        if wandb.run:
            wandb.log({
                'Episode': episode,
                'Step': step,
                'Min Return': np.round(np.min(self.ep_rewards[-self.verbosity:]), 3),
                'Max Return': np.round(np.max(self.ep_rewards[-self.verbosity:]), 3),
                'Mean Reward': mean_ep_reward,
                'Mean Length': mean_ep_length,
                'Mean Loss': mean_ep_loss,
                'Time Delta': time_since_last_record
            })

        # Log metrics to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Episode/Min Return', np.round(np.min(self.ep_rewards[-self.verbosity:]), 3), episode)
            self.tensorboard_writer.add_scalar('Episode/Max Return', np.round(np.max(self.ep_rewards[-self.verbosity:]), 3), episode)
            self.tensorboard_writer.add_scalar('Episode/Mean Reward', mean_ep_reward, episode)
            self.tensorboard_writer.add_scalar('Episode/Mean Length', mean_ep_length, episode)
            self.tensorboard_writer.add_scalar('Episode/Mean Loss', mean_ep_loss, episode)
            self.tensorboard_writer.add_scalar('Episode/Time Delta', time_since_last_record, episode)

        if verbose:
            logging.info(
                f"Episode {episode:>7} - "
                f"Step {step:>10} - "
                f"Min Return {np.round(np.min(self.ep_rewards[-self.verbosity:]), 3):>8.3f} - "
                f"Max Return {np.round(np.max(self.ep_rewards[-self.verbosity:]), 3):>8.3f} - "
                f"Mean Reward {mean_ep_reward:>10.3f} - "
                f"Mean Length {mean_ep_length:>10.3f} - "
                f"Mean Loss {mean_ep_loss:>8.3f} - "
                f"Time Delta {time_since_last_record:>9.3f}"
            )

        with self.save_log.open("a") as f:
            f.write(
                f"{episode:8d}{step:8d}"
                f"{np.round(np.min(self.ep_rewards[-self.verbosity:]), 3):15.3f}{np.round(np.max(self.ep_rewards[-self.verbosity:]), 3):15.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

    def close(self):
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        if wandb.run:
            wandb.finish()
