import pathlib
import numpy as np
import matplotlib.pyplot as plt
from ddpg.tools import OrnsteinUhlenbeckProcess

from ddpg.ddpg import Agent
from water_filter import water_filter


def main():
    run = 3
    pathlib.Path('./trained_models/run' + str(run)).mkdir(parents=True, exist_ok=True)
    path = './trained_models/run' + str(run) + '/'

    """ Init filter model"""
    k_1 = 0.1
    k_2 = 0.05
    delta = 1
    s_1 = 1/2
    s_2 = 1/2
    filter_system = water_filter.Filter(s_1, s_2, k_1, k_2, delta)

    """ Init agents """
    agent1 = Agent(agent_index=1, save_dir=path)
    agent2 = Agent(agent_index=2, save_dir=path)

    trajectory_length = 1000
    epochs = 100

    reward_epochs = []
    for epoch in range(epochs):
        # Reset
        time = 0
        s_1 = np.random.uniform()
        s_2 = np.random.uniform()
        filter_system.state_1 = s_1
        filter_system.state_2 = s_2
        noise1 = OrnsteinUhlenbeckProcess(size=1)
        noise2 = OrnsteinUhlenbeckProcess(size=1)
        state = filter_system.state()

        for n in range(trajectory_length):
            a_1 = np.clip(agent1.pick_action(state)[0] + noise1.generate(time), 0, 1)[0]
            a_2 = np.clip(agent2.pick_action(state)[1] + noise2.generate(time), 0, 1)[0]
            action = np.array([a_1, a_2])
            reward = filter_system.step(a_1, a_2)
            next_state = filter_system.state()

            agent1.remember(state, action, reward, next_state, 0)
            agent1.train()

            agent2.remember(state, action, reward, next_state, 0)
            agent2.train()

            if (epoch > 0 or n > 64) and np.random.rand() < 0.1:
                agent1.actor.tower_21.set_weights(agent2.actor.tower_21.get_weights())
                agent1.actor.tower_22.set_weights(agent2.actor.tower_22.get_weights())
            if (epoch > 0 or n > 64) and np.random.rand() < 0.1:
                agent2.actor.tower_11.set_weights(agent1.actor.tower_11.get_weights())
                agent2.actor.tower_12.set_weights(agent1.actor.tower_12.get_weights())

            state = next_state
            time += 1

        # Evaluate current model over 10 additional epochs without exploration
        evaluation_epochs = 10
        cum_reward = 0
        for eva_e in range(evaluation_epochs):
            s_1 = np.random.uniform()
            s_2 = np.random.uniform()
            filter_system.state_1 = s_1
            filter_system.state_2 = s_2
            state = filter_system.state()
            for n in range(trajectory_length):
                a_1 = agent1.pick_action(state)[0]
                a_2 = agent2.pick_action(state)[1]
                cum_reward += filter_system.step(a_1, a_2)
                state = filter_system.state()
        reward_epochs.append(cum_reward / (trajectory_length*evaluation_epochs))
        print('Epoch: ' + str(epoch) + ', Avg_reward: ' + "{:.3f}".format(reward_epochs[-1]))

    ''' Save models and avg_reward trajectory'''
    agent1.save_models()
    agent2.save_models()
    np.save(path + 'avg_reward', np.array(reward_epochs))

    ''' Plot reward trajectory'''
    plt.figure()
    plt.plot(reward_epochs)
    plt.xlabel('epoch')
    plt.ylabel('$r_{avg}$')
    plt.show()


if __name__ == "__main__":
    main()
