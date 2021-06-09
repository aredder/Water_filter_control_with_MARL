import os
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.initializers import RandomUniform


class CriticNetwork(keras.Model):
    def __init__(self, agent_index, save_dir, layer1_dims=512, name='critic'):
        super(CriticNetwork, self).__init__()

        self.agent_index = agent_index
        self.layer1_dims = layer1_dims

        self.save_dir = save_dir
        self.model_name = name + str(self.agent_index)
        self.save_file = os.path.join(self.save_dir, self.model_name + '_ddpg.tf')

        self.layer1 = Dense(self.layer1_dims, activation=LeakyReLU(alpha=0.1))
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.layer1(tf.concat([state, action], axis=1))

        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, agent_index, save_dir, action_space=2, name='actor'):
        super(ActorNetwork, self).__init__()

        self.agent_index = agent_index
        self.action_space = action_space

        self.save_dir = save_dir
        self.model_name = name + str(self.agent_index)
        self.save_file = os.path.join(self.save_dir, self.model_name + '_ddpg.tf')

        self.tower_11 = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.tower_21 = Dense(128, activation=LeakyReLU(alpha=0.1))

        self.tower_12 = Dense(1, activation='sigmoid',
                              kernel_initializer=RandomUniform(minval=-10**(-1), maxval=10**(-1)))
        self.tower_22 = Dense(1, activation='sigmoid',
                              kernel_initializer=RandomUniform(minval=-10**(-1), maxval=10**(-1)))

        # Set mu_2 to be not trainable for agent 1 and vise versa.
        if self.agent_index == 1:
            self.tower_21.trainable = False
            self.tower_22.trainable = False
        else:
            self.tower_11.trainable = False
            self.tower_12.trainable = False

    def call(self, state):
        prob1 = self.tower_11(state)
        prob2 = self.tower_21(state)

        prob1 = self.tower_12(prob1)
        prob2 = self.tower_22(prob2)

        action = tf.concat([prob1, prob2], axis=1)
        return action

