
import os
import time
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from rich.progress import Progress

from src.agent import Agent
from src.metrics import MetricLogger
from src.utils import (
    stats_printer, 
    test_stats, 
    save_3d_array_to_xlsx
)


class Engine:
    def __init__(
        self, 
        env: gym.Env,
        agent: Agent,
        logger: MetricLogger,
        en_train: bool = False,
        en_eval: bool = False,
        en_visual: bool = False,
        episodes: int = 1000,
        test_cases: int = 10,
        verbosity: int = 10,
        workload: list = None
    ):
        self.env = env
        self.visual = en_visual
        self.agent = agent
        self.logger = logger
        self.episodes = episodes
        self.test_cases = test_cases
        self.verbosity = verbosity
        self.en_train = en_train
        self.en_eval = en_eval

        self.last_ep = 0

        if self.en_train:
            self.train()

        if self.en_eval:
            self.eval()

    def train(self):
        # Agent training loop
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=self.episodes)
        
            # Agent training loop
            for e in range(self.episodes):

                # Initialize environment
                state, info = self.env.reset()

                # Reset end-of-episode flag
                terminated, truncated = False, False

                # Train agent
                while not terminated and not truncated:

                    # 0. Show environment (the visual) [WIP]
                    if self.visual:
                        self.env.render()

                    # 1. Run agent on the state
                    action = self.agent.act(state)

                    # 2. Agent performs action
                    next_state, reward, terminated, truncated, info = self.env.step(
                        action)

                    # 3. Monitor environment
                    # stats_printer(info)

                    # 4. Remember
                    self.agent.cache(state, next_state, action, reward, terminated)

                    # 5. Learn
                    loss = self.agent.learn(e)

                    # 6. Logging
                    self.logger.log_step(reward, loss)

                    # 7. Update state
                    state = next_state

                    # 8. Calculate advance
                    advance = 1 if self.last_ep != e else 0

                    # 9. Update the progress bar description
                    progress.update(task, advance=advance, description=self.logger.fetch(episode=e, step=self.agent.curr_step))

                    # 10. Update last episode
                    self.last_ep = e
                    
                self.logger.log_episode()

                if e % self.verbosity == 0 and e > 0:
                    self.logger.record(episode=e, step=self.agent.curr_step)
            
            self.env.close()
            self.logger.close()
    
    def eval(self):
        # Initialize action trajectory
        action_trajectory = []

        positive_reward = []
        mission_status = []

        # Check demo value in config
        if self.env.demo == 0:
            print("Demo value in config is not set to True. Please set it to True to run evaluation.")
            return

        # Agent evaluation loop
        for test_case in range(self.test_cases):

            state, info = self.env.reset()

            terminated, truncated = False, False
            score = 0

            start = time.time()

            while not terminated and not truncated:
                
                # 0. Show environment (the visual) [WIP]
                if self.visual:
                    self.env.render()

                # 1. Run agent on the state
                action = self.agent.act(state)

                # 2. Agent performs action
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # 3. Monitor environment
                # stats_printer(info)

                # 4. Update state
                state = next_state

                # 5. Accumulate last score
                score += reward

                # 6. Append action to trajectory
                action_trajectory.append(action)

                # 7. Update success in acquiring a positive reward
                positive_reward.append(True if reward > 0.0 else False)

                # 8. Update mission status
                if terminated:
                    mission_status.append(True if score > 100 else False)

            end = time.time()
            print(f"Test case {test_case} took {end - start} seconds")

        self.env.close()
        self.logger.close()

        # Print test stats
        print(action_trajectory)

        # Output test results
        test_stats(positive_reward, mission_status)
