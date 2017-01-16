import os
import gym
import argparse
import numpy as np
import tensorflow as tf
from random import shuffle
from parallelenvs import ParallelEnvs
from estimators import PolicyNet, ValueNet, summary_writer_op

# Create an argument parser (for terminal integration)
parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('env_name', type=str, help='Gym environment name')
parser.add_argument('num_episodes', type=int, help='Number of training episodes')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate used when performing gradient descent (default: 3e-4)')
parser.add_argument('num_workers', type=int, default=4,
                    help='Number of pararell environments')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of episodes before training')
parser.add_argument('--render', type=str, choices=['Y', 'N'], default='N',
                    help='Should environment be rendered?')
parser.add_argument('--logdir', type=str, default='summaries',
                    help='Directory to save tensorboard summaries')
args = parser.parse_args()
env = gym.make(args.env_name)
num_actions = env.action_space.n
num_features = env.observation_space.shape[0]
env.close()
logdir = os.path.join(args.logdir, args.env_name)
#env_name = 'LunarLander-v2'
#render = True
#logdir = 'summaries'
#num_episodes = 1000
#learning_rate = 5e-3
#num_workers = 4
#batch_size = 16

# Create estimators
tf.reset_default_graph()
policy_net = PolicyNet(learning_rate=args.learning_rate,
                       num_actions=num_actions,
                       num_features=num_features)
value_net = ValueNet(learning_rate=args.learning_rate, num_features=num_features)

with tf.Session() as sess:
    # Create envs
    envs = ParallelEnvs(env_name=args.env_name,
                        policy_net=policy_net,
                        num_actions=num_actions,
                        sess=sess,
                        render=args.render)
    envs.create_workers(args.num_workers)
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Create summary writer
    summary_writer = summary_writer_op(sess=sess,
                                       logdir=logdir,
                                       policy_net=policy_net,
                                       value_net=value_net)

    print('Started training...')
    for i_episode in range(1, args.num_episodes + 1):
        experience = envs.run(args.batch_size)
        # Shuffle experience
        shuffle(experience)
        states, actions, rewards, returns = zip(*experience)

        # Calculate value of states (baseline)
        states_value = value_net.predict(sess, states)
        value_errors = returns - np.squeeze(states_value)

        # Update baseline weights
        value_net.update(sess, states, value_errors)
        #Update policy weights
        policy_net.update(sess, states, actions, value_errors)

        # Write summaries
        avg_reward = np.sum(rewards) / args.batch_size
        summary_writer(states, actions, value_errors, returns, avg_reward, avg_reward)
        # Print information
        print('Episode {}/{} | Avg_reward: {}'.format(i_episode, args.num_episodes, avg_reward))
