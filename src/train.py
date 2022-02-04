
import gc
import torch

from src.core import *
from src.functions import *


def train(env, daedalous, logger):

    # Agent training loop
    for e in range(EPISODES):

        # Initialize environment
        state = env.reset()

        # Reset end-of-episode flag
        done = False

        # Train daedalous
        while not done:

            # 1. Show environment (the visual) [WIP]
            # env.render()

            # 2. Run agent on the state
            action = daedalous.act(state)

            # 3. Agent performs action
            next_state, reward, done, info = env.step(action)

            # 4. Monitor environment
            # stats_printer(info)

            # 5. Remember
            daedalous.cache(state, next_state, action, reward, done)

            # 6. Learn
            loss = daedalous.learn(e)

            # 7. Logging
            logger.log_step(reward, loss)

            # 8. Update state
            state = next_state

        logger.log_episode()

        if e % VERBOSITY == 0 and e > 0:
            logger.record(episode=e, step=daedalous.curr_step)

        torch.cuda.empty_cache()
        gc.collect()

    # env.close()
