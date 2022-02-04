
from src.core import *
from src.functions import *


def test(env, daedalous):

    positive_reward = []
    mission_status = []

    for test_case in range(TEST_SET_CARDINALITY):

        state = env.reset()

        done = False
        score = 0

        while not done:
            # 1. Show environment (the visual) [WIP]
            env.render()

            # 2. Run agent on the state
            action = daedalous.act(state)

            # 3. Agent performs action
            next_state, reward, done, info = env.step(action)

            # 4. Monitor environment
            # stats_printer(info)

            # 5. Update state
            state = next_state

            # 6. Accumulate last score
            score += reward

            # 7. Update success in acquiring a positive reward
            positive_reward.append(True if reward > 0.0 else False)

            # 8. Update mission status
            if done:
                mission_status.append(True if score > 100 else False)

    # env.close()

    # Output test results
    # test_stats(positive_reward, mission_status)
