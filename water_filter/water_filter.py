import numpy as np
from scipy.integrate import odeint


class Filter:
    def __init__(self, flow_init, water_init, k_1, k_2, delta):
        self.state_1 = flow_init
        self.state_2 = water_init
        self.delta = delta
        self.a_1 = None
        self.a_2 = None
        self.k_1 = k_1
        self.k_2 = k_2

    def model(self, y, t):
        dydt = - self.k_2*self.a_2*np.sqrt(t) + self.k_1*self.a_1*self.state_1
        return dydt

    def state(self):
        return np.array([self.state_1, self.state_2])

    def reward(self, a_1):
        if self.state_2 == 0 or self.state_2 == 1:
            reward = self.state_1 * a_1 / 2
        else:
            reward = (self.state_1 * a_1 +
                      np.exp((2 * self.state_2 - 1) ** 2 / (10 * (self.state_2 - 1) * self.state_2)))/2
        return reward

    def step(self, a_1, a_2):
        # step filter
        self.a_1 = a_1
        self.a_2 = a_2
        y0 = self.state_2
        t = np.array([0, self.delta])
        # solve ODE for t = delta
        self.state_2 = odeint(self.model, y0, t)[1][0]
        if self.state_2 > 1:
            self.state_2 = 1

        # step main flow
        self.state_1 += 0.05 * np.random.choice([-1, 0, 1])
        if self.state_1 > 1:
            self.state_1 = 2 - self.state_1
        elif self.state_1 < 0:
            self.state_1 = - self.state_1

        return self.reward(a_1)
