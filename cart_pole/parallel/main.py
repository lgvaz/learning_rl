import numpy as np
import tensorflow as tf
from random import shuffle
from parallelenvs import ParallelEnvs
from estimators import PolicyNet, ValueNet, summary_writer_op

log_dir = 'summaries'
num_episodes = 1000
num_actions = 2
learning_rate = 5e-3
num_workers = 8
batch_size = 16
# Create estimators
tf.reset_default_graph()
policy_net = PolicyNet(learning_rate=learning_rate, num_actions=2)
value_net = ValueNet(learning_rate=learning_rate)

with tf.Session() as sess:
    # Create envs
    envs = ParallelEnvs(env_name='CartPole-v0',
                        policy_net=policy_net,
                        num_actions=num_actions,
                        sess=sess,
                        render=True)
    envs.create_workers(num_workers)
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Create summary writer
    summary_writer = summary_writer_op(sess=sess,
                                       logdir=log_dir,
                                       policy_net=policy_net,
                                       value_net=value_net)

    print('Started training...')
    for i_episode in range(1, num_episodes + 1):
        experience = envs.run(batch_size)
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
        avg_reward = np.sum(rewards) / batch_size
        summary_writer(states, actions, value_errors, returns, avg_reward, avg_reward)
        # Print information
        print('Episode {}/{} | Avg_reward: {}'.format(i_episode, num_episodes, avg_reward))
