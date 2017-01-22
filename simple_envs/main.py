import os
import gym
import argparse
import itertools
import numpy as np
import tensorflow as tf
from memory import ReplayMemory
from estimators import QNet, copy_vars, egreedy_policy

env_name = 'LunarLander-v2'
num_episodes = 100000
max_replays = 200000
min_replays = 100000
learning_rate = 3e-4
clip_grads = True
logdir = os.path.join(env_name, 'summaries')
epsilon_min = 0.1
epsilon_max = 1
batch_size = 32
stop_exploration = 10000
discount_factor = 0.99
target_update_frequency = 10000

parser = argparse.ArgumentParser(description=(
                                 'Run training episodes, periodically saves the model, '
                                 'and creates a tensorboard summary.'))
parser.add_argument('env_name', type=str, help='Gym environment name')
parser.add_argument('num_episodes', type=int, help='Number of training episodes')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate used when performing gradient descent (default: 3e-4)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Split episodes in batchs of specified size')
parser.add_argument('--epsilon_max', type=float, default=1,
                    help='The maximum exploration rate')
parser.add_argument('--epsilon_min', type=float, default=0.1,
                    help='The minimum exploration rate')
# TODO: Maybe this should be a positional arg? (Required arg)
parser.add_argument('--exploration_stop', type=int, default=1000,
                    help='Steps before epsilon reaches minimum')
parser.add_argument('--max_replays', type=int, default=200000,
                    help='Maximum number of replays')
parser.add_argument('--min_replays', type=int, default=100000,
                    help='Number os replays generate by random agent before start training')
parser.add_argument('--number_steps_limit', type=int, default=1500,
                    help='Number of maximum steps allowed per episode')
env = gym.make(env_name)
num_actions = env.action_space.n
num_features = env.observation_space.shape[0]

# Create experience replay
replays = ReplayMemory(max_replays, num_features)
# Populate replay memory with random actions
print('Populating replays...')
state = env.reset()
for i_replay in range(min_replays):
    # Pick an action
    action = np.random.choice(np.arange(num_actions))
    next_state, reward, done, _ = env.step(action)
    # Save replay
    replays.add_replay(state, action, reward, done)
    # Update state
    print(i_replay)
    if done:
        done = False
        state = env.reset()
    else:
        state = next_state

# Create network
main_qnet = QNet(num_features=num_features,
            num_actions=num_actions,
            learning_rate=learning_rate,
            scope='main',
            clip_grads=clip_grads,
            create_summary=True)
target_qnet = QNet(num_features=num_features,
            num_actions=num_actions,
            learning_rate=learning_rate,
            scope='target',
            clip_grads=clip_grads,
            create_summary=False)

# Calculate epsilon step size
epsilon_step = -np.log(epsilon_min) / stop_exploration

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Create the update target operation
    target_update = copy_vars(sess=sess, from_scope='main', to_scope='target')
    target_update()
    # Create summary writer
    summary_writer = main_qnet.create_summary_op(sess, logdir)

    print('Started training...')
    num_steps = 0
    for i_episode in range(num_episodes):
            # Restart env
            state = env.reset()
            ep_reward = 0

            for i_step in itertools.count():
                # Exponentially decay epsilon
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-epsilon_step * i_episode)
                # Choose an action
                action_values = main_qnet.predict(sess, state[np.newaxis])
                action_probs = egreedy_policy(action_values, epsilon)
                action = np.random.choice(np.arange(num_actions), p=action_probs)
                # Do the action
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                # Save experience
                replays.add_replay(state, action, reward, done)

                # Sample a batch to train
                b_states, b_next_states, b_actions, b_rewards, b_done = replays.sample(batch_size)
                # Calculate next max Q
                b_value_next_states = target_qnet.predict(sess, b_next_states)
                b_next_qmax = np.max(b_value_next_states, axis=1)
                # Calculate TD targets
                b_targets = b_rewards + (1 - b_done) * discount_factor * b_next_qmax

                # Update main network weigths
                main_qnet.update(sess, b_states, b_actions, b_targets)

                # Update target network
                num_steps += 1
                if num_steps % target_update_frequency == 0:
                    print('Updating target network...')
                    target_update()

                # Update state
                if done or i_step == 1500:
                    break
                else:
                    state = next_state

            # # Write summaries
            # summary_writer(states=b_states,
            #                actions=b_actions,
            #                targets=b_targets,
            #                reward=ep_reward,
            #                length=i_step,
            #                epsilon=epsilon)
            # Show information
            print('Episode {}/{}'.format(i_episode, num_episodes), end=' | ')
            print('Epsilon: {:.3f}'.format(epsilon), end=' | ')
            print('Reward: {:.3f}'.format(ep_reward), end=' | ')
            print('Length: {}'.format(i_step))
