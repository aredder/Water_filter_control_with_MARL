import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ddpg.ddpg import Agent
from water_filter import water_filter

plt.rcParams['animation.ffmpeg_path'] ='C:/Program Files/ImageMagick-7.0.11-Q8/ffmpeg.exe'

animate_waterlevel = True


def evaluation(animate_waterlevel=True):

    run = 2
    path = './trained_models/run' + str(run) + '/'
    agent1 = Agent(agent_index=1, save_dir=path)
    agent2 = Agent(agent_index=2, save_dir=path)

    agent1.load_models()
    agent2.load_models()

    trajectory_length = 1000

    """ Plot avg_reward """
    reward_epochs = np.load(path + 'avg_reward.npy')
    plt.figure()
    plt.plot(reward_epochs)
    plt.xlabel('epoch')
    plt.ylabel('$r_{avg}$')
    plt.savefig('visualisations/avg_reward.png')

    """ Init filter model"""
    k_1 = 0.1
    k_2 = 0.05
    delta = 1
    s_1 = 1/2
    s_2 = 1/2
    filter_system = water_filter.Filter(s_1, s_2, k_1, k_2, delta)

    """ Sample trajectory"""
    main_flow_traj = []
    inflow_traj = []
    water_level_traj = []
    action1_traj = []
    action2_traj = []
    filter_system.state_1 = s_1
    filter_system.state_2 = s_2
    state = filter_system.state()

    for n in range(trajectory_length):
        main_flow_traj.append(filter_system.state_1)
        water_level_traj.append(filter_system.state_2)
        a_1 = agent1.pick_action(state)[0]
        a_2 = agent2.pick_action(state)[1]
        #a_1 = 1/2
        #a_2 = 1/2
        inflow_traj.append(filter_system.state_1*a_1*k_1)
        action1_traj.append(a_1)
        action2_traj.append(a_2)
        _ = filter_system.step(a_1, a_2)
        state = filter_system.state()

    """ Flow trajectories """
    plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(main_flow_traj)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('$s_{1,n}$')
    plt.title('Main flow')
    plt.ylim(-0.01, 1.01)

    ax2 = plt.subplot(312)
    plt.plot(inflow_traj)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('$f_n$')
    plt.ylim(-0.01, 0.13)
    plt.title('Inflow')

    ax3 = plt.subplot(313, sharex=ax2)
    plt.plot(water_level_traj)
    plt.xlabel('n')
    plt.ylabel('$s_{2,n}$')
    plt.ylim(-0.01, 1.01)
    plt.title('Water level')
    plt.savefig('visualisations/water_traj.png')

    """ Action trajectories """
    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(action1_traj)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('$a_{1,n}$')
    plt.ylim(-0.01, 1.01)
    plt.title('Inflow valve')
    plt.text(750, 1.04, '0: Closed, 1: Open')

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(action2_traj)
    plt.xlabel('n')
    plt.ylabel('$a_{2,n}$')
    plt.ylim(-0.01, 1.01)
    plt.title('Outflow valve')
    plt.text(750, 1.04, '0: Closed, 1: Open')
    plt.savefig('visualisations/action.png')

    if animate_waterlevel:
        """ Animate water level """
        y = water_level_traj
        fig = plt.figure(figsize=(5, 5))
        plt.ylabel('Water level')
        ax = plt.axes(xlim=(0, 1), ylim=(0, 1), aspect='equal')

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        def animate(i):
            ax.fill_between(x=(0 + np.exp(-7), 1 - np.exp(-7)), y1=0, y2=1, color='white')
            rect = ax.fill_between(x=(0 + np.exp(-7), 1 - np.exp(-7)), y1=0, y2=y[i], color='blue')
            return rect,

        anim = animation.FuncAnimation(fig, animate, frames=len(y), interval=1, blit=True, repeat=False,
                                       save_count=1000)
        writer = animation.FFMpegWriter(fps=15)
        anim.save("./visualisations/animation_waterlevel.mp4", writer=writer)


evaluation(animate_waterlevel)
