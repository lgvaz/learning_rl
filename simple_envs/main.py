import os
import gym
from gym import wrappers
import argparse
import itertools
import numpy as np
import tensorflow as tf
from memory import ReplayMemory
from estimators import QNet, copy_vars, egreedy_policy


parser = argparse.ArgumentParser(description=(
                                 'Run training episodes, periodically saves the model, '
                                 'and creates a tensorboard summary.'))
parser.add_argument('env_name', type=str, help='Gym environment name')
parser.add_argument('num_episodes', type=int, help='Number of training episodes')
parser.add_argument('stop_exploration', type=int,
                    help='Steps before epsilon reaches minimum')
parser.add_argument('target_update_frequency', type=int,
                    help='Update target network every "n" steps')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate used when performing gradient descent (default: 3e-4)')
# TODO: improve help text
parser.add_argument('--double_learning', type=str, choices=['Y', 'N'], default='N',
                    help='Whether double q-learning should be used or not (default=N)')
parser.add_argument('--clip_grads', type=str, choices=['Y', 'N'], default='Y',
                    help='Whether the grads should be clipped or not (default=Y)')
parser.add_argument('--discount_factor', type=float, default=0.99,
                    help='How much the agent should look into the future (default=0.99)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Split episodes in batchs of specified size (default=32)')
parser.add_argument('--epsilon_max', type=float, default=1,
                    help='The maximum exploration rate (default=1)')
parser.add_argument('--epsilon_min', type=float, default=0.1,
                    help='The minimum exploration rate (default=0.1)')
parser.add_argument('--max_replays', type=int, default=200000,
                    help='Maximum number of replays (default=200000)')
parser.add_argument('--min_replays', type=int, default=100000,
                    help='Number os replays generate by random agent before start training (default=100000)')
parser.add_argument('--number_steps_limit', type=int, default=1500,
                    help='Number of maximum steps allowed per episode (default=1500)')
args = parser.parse_args()

# Ask experiment name
exp_name = input('Name of experiment: ')
envdir = os.path.join('experiments', args.env_name)
argsdir = os.path.join(envdir, 'args')
logdir = os.path.join(envdir, 'summaries', exp_name)
savedir = os.path.join(envdir, 'checkpoints', exp_name)
videodir = os.path.join(envdir, 'videos', exp_name)
# Create checkpoint directory
if not os.path.exists(savedir):
    os.makedirs(savedir)
savepath = os.path.join(savedir, 'graph.ckpt')
# Create videos directory
if not os.path.exists(videodir):
    os.makedirs(videodir)
# Save args to a file
if not os.path.exists(argsdir):
    os.makedirs(argsdir)
argspath = os.path.join(argsdir, exp_name) + '.txt'
with open(argspath, 'w') as f:
    for arg, value in args.__dict__.items():
        f.write(': '.join([str(arg), str(value)]))
        f.write('\n')

# Create env
env = gym.make(args.env_name)
num_actions = env.action_space.n
num_features = env.observation_space.shape[0]

# Create experience replay
replays = ReplayMemory(args.max_replays, num_features)
# Populate replay memory with random actions
print('Populating replays...')
state = env.reset()
for i_replay in range(args.min_replays):
    # Pick an action
    action = np.random.choice(np.arange(num_actions))
    next_state, reward, done, _ = env.step(action)
    # Save replay
    replays.add_replay(state, action, reward, done)
    # Update state
    if done:
        print('\rReplay {}/{}'.format(i_replay, args.min_replays), end='', flush=True)
        done = False
        state = env.reset()
    else:
        state = next_state
print()

# Create network
main_qnet = QNet(num_features=num_features,
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            scope='main',
            clip_grads=args.clip_grads,
            create_summary=True)
target_qnet = QNet(num_features=num_features,
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            scope='target',
            clip_grads=args.clip_grads,
            create_summary=False)

# Calculate epsilon step size
epsilon_step = -np.log(args.epsilon_min) / args.stop_exploration
# Configure the maximum number of steps
env.spec.tags.update({'wrapper_config.TimeLimit.max_episode_steps': args.number_steps_limit})
# Create monitor for recording episodes
env = wrappers.Monitor(env=env, directory=videodir, force=True)


with tf.Session() as sess:
    # Create tensorflow saver
    saver = tf.train.Saver()
    # Verify if a checkpoint already exists
    latest_checkpoint = tf.train.latest_checkpoint(savedir)
    if latest_checkpoint is not None:
        print('Loading latest checkpoint...')
        saver.restore(sess, latest_checkpoint)
    else:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
    # Create the update target operation
    target_update = copy_vars(sess=sess, from_scope='main', to_scope='target')
    target_update()
    # Create summary writer
    summary_writer = main_qnet.create_summary_op(sess, logdir)

    print('Started training...')
    num_steps = 0
    global_step = 0
    for i_episode in range(args.num_episodes):
            # Restart env
            state = env.reset()
            ep_reward = 0

            for i_step in itertools.count():
                # Exponentially decay epsilon
                epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) \
                          * np.exp(-epsilon_step * global_step)
                global_step += 1
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
                b_states, b_next_states, b_actions, b_rewards, b_done = replays.sample(args.batch_size)
                # Calculate best action value using simple q-learning
                if args.double_learning == 'N':
                    # Calculate next max Q
                    b_value_next_states = target_qnet.predict(sess, b_next_states)
                    b_next_qmax = np.max(b_value_next_states, axis=1)
                # Calculate best action value using double q-learning
                if args.double_learning == 'Y':
                    b_value_next_states = main_qnet.predict(sess, b_next_states)
                    b_next_action = np.argmax(b_value_next_states, axis=1)
                    b_next_q = target_qnet.predict(sess, b_next_states)
                    b_next_qmax = b_next_q[np.arange(args.batch_size), b_next_action]
                # Calculate TD targets
                b_targets = b_rewards + (1 - b_done) * args.discount_factor * b_next_qmax

                # Update main network weigths
                main_qnet.update(sess, b_states, b_actions, b_targets)

                # Update target network
                num_steps += 1
                if num_steps % args.target_update_frequency == 0:
                    print('Updating target network...')
                    target_update()

                # Update state
                if done:
                    break
                else:
                    state = next_state

            # Write summaries
            summary_writer(states=b_states,
                           actions=b_actions,
                           targets=b_targets,
                           reward=ep_reward,
                           length=i_step,
                           epsilon=epsilon)
            # Save model
            if i_episode % 100 == 0:
                print('Saving model to: {}'.format(savedir))
                saver.save(sess, savepath)

            # Show information
            print('Episode {}/{}'.format(i_episode, args.num_episodes), end=' | ')
            print('Epsilon: {:.3f}'.format(epsilon), end=' | ')
            print('Reward: {:.3f}'.format(ep_reward), end=' | ')
            print('Length: {}'.format(i_step))
