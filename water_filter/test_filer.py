import numpy as np
import matplotlib.pyplot as plt

from water_filter import Filter

r_1 = 0.1
r_2 = 1/200000
delta = 1
s_1 = 1/2
s_2 = 1/2

filter_system = Filter(s_1, s_2, r_1, r_2, delta)

main_flow_traj = []
inflow_traj = []
water_level_traj = []
action1_traj = []
action2_traj = []
s_1 = np.random.uniform()
s_2 = np.random.uniform()
filter_system.state_1 = s_1
filter_system.state_2 = s_2
filter_system.walk = s_2
state = filter_system.state()
for n in range(1000):
    main_flow_traj.append(filter_system.state_1)
    water_level_traj.append(filter_system.state_2)
    a_1 = 0.5
    a_2 = 0.5
    inflow_traj.append(filter_system.state_1 * a_1 * r_1)
    action1_traj.append(a_1)
    action2_traj.append(a_2)
    _ = filter_system.step(a_1, a_2)
    state = filter_system.state()

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
plt.ylim(-0.01, 0.11)
plt.title('Inflow')

ax3 = plt.subplot(313, sharex=ax2)
plt.plot(water_level_traj)
plt.xlabel('n')
plt.ylabel('$s_{2,n}$')
plt.ylim(-0.01, 1.01)
plt.title('Water level')
plt.savefig('water_traj.png')

plt.show()