import gym
import numpy as np
import tensorflow as tf
from worker import Worker
from estimators import *

env_name = 'LunarLander-v2'
env = gym.make(env_name)
num_actions = env.action_space.n
num_features = env.observation_space.shape[0]
learning_rate = 0.002
num_workers = 8
num_steps = 1000000
stop_exploration = 300000
discount_factor = 0.99
online_update_step = 5
target_update_step = 6000

# Create the shared networks
online_net = QNet(num_features, num_actions, learning_rate,
                  scope='online', clip_grads='N')
target_net = QNet(num_features, num_actions, learning_rate,
                  scope='target', clip_grads='N')

# Create tensorflow coordinator to manage when threads should stop
coord = tf.train.Coordinator()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    workers = Worker(env_name, num_actions, num_workers, num_steps, stop_exploration, discount_factor, online_update_step, target_update_step, online_net, target_net, sess, coord)
    workers.run()
