import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class PolicyNet:
    '''
    Creates a neural network that approximates
    actions probabilities given a state

    Args:
        learning_rate: The learning rate used by the optmizer
        num_actions: Number of valid actions
    '''
    def __init__(self, learning_rate, num_actions):
        # Placeholders
        self.states = tf.placeholder(name='states',
                                     shape=(None, 4),
                                     dtype=tf.float32)
        self.returns = tf.placeholder(name='returns',
                                      shape=(None),
                                      dtype=tf.float32)
        self.actions = tf.placeholder(name='chosen_action',
                                      shape=(None),
                                      dtype=tf.int32)

        with tf.variable_scope('policy'):
            self.fc = slim.fully_connected(self.states, 16)
            # Final/output layer
            self.output = slim.fully_connected(self.fc,
                                               num_actions,
                                               activation_fn=tf.nn.softmax)

        # Optimization process (to increase likelihood of a good action)
        batch_size = tf.shape(self.states)[0]
        # Select the ids of picked actions
        # action_ids = (i_batch * NUM_ACTIONS) + action
        action_ids = tf.range(batch_size) * tf.shape(self.output)[1] + self.actions
        # Select probability of chosen actions
        chosen_actions = tf.gather(tf.reshape(self.output, [-1]), action_ids)
        eligibility = tf.log(chosen_actions)
        # Change the likelihood of taken action using the return (self.returns)
        self.loss = - tf.reduce_mean(self.returns * eligibility)
        opt = tf.train.AdamOptimizer(learning_rate)
        # We should perform gradient ascent in the likelihood of specified action
        # which is the same as performing gradient descent on the negative of the loss
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy')
        grads_and_vars = opt.compute_gradients(self.loss, local_vars)
        self.global_step = slim.get_or_create_global_step()
        self.train_op = opt.apply_gradients(grads_and_vars, self.global_step)

        # Add summaries
        tf.summary.histogram('last_hidden', self.fc)
        tf.summary.histogram('action_probs', self.output)
        tf.summary.scalar('policy_loss', self.loss)

    def predict(self, sess, states):
        '''
        Calculate action probabilities for given state

        Args:
            sess: The current tensorflow session
            states: Observation of the state(s)

        Returns:
            Actions probabilities
        '''
        return sess.run(self.output, feed_dict={self.states: states})

    def update(self, sess, states, actions, returns):
        feed_dict = {self.states: states,
                     self.actions: actions,
                     self.returns: returns}
        sess.run(self.train_op, feed_dict=feed_dict)


class ValueNet:
    '''
    Creates a neural network to approximates states value

    Args:
        learning_rate: The learning rate used by the optimizer
    '''
    def __init__(self, learning_rate):
        # Placeholders
        self.states = tf.placeholder(name='states',
                                     shape=(None, 4),
                                     dtype=tf.float32)
        # TD targets
        self.targets = tf.placeholder(name='targets',
                                      shape=(None),
                                      dtype=tf.float32)

        # Final/output layer
        with tf.variable_scope('value_net'):
            self.fc = slim.fully_connected(self.states, 16)
            self.output = slim.fully_connected(inputs=self.fc,
                                               num_outputs=1,
                                               activation_fn=None)

        # Loss (mean squared error)
        self.loss = tf.reduce_mean(tf.squared_difference(self.targets, self.output))
        opt = tf.train.AdamOptimizer(learning_rate)
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value_net')
        grads_and_vars = opt.compute_gradients(self.loss, local_vars)
        self.global_step = slim.get_or_create_global_step()
        self.train_op = opt.apply_gradients(grads_and_vars)

        # Add summaries
        tf.summary.histogram('state_values', self.output)
        tf.summary.scalar('average_state_value', tf.reduce_mean(self.output))
        tf.summary.scalar('max_state_value', tf.reduce_max(self.output))
        tf.summary.scalar('baseline_loss', self.loss)

    def predict(self, sess, states):
        '''
        Calculate the value of the current state

        Args:
            sess: The current tensorflow session
            states: Observation of the state(s)

        Returns:
            Value of the current state
        '''
        return sess.run(self.output, feed_dict={self.states: states})

    def update(self, sess, states, targets):
        feed_dict = {self.states: states,
                     self.targets: targets}
        sess.run(self.train_op, feed_dict=feed_dict)


def test_updates(learning_rate, num_actions, batch_size=100):
    ''' Test if the weigth updates are giving the desired outputs '''
    # Create a new policy
    tf.reset_default_graph()
    policy = PolicyNet(learning_rate=learning_rate,
                       num_actions=num_actions)
    baseline = ValueNet(learning_rate=learning_rate)
    # Generate states
    state = np.random.random((batch_size, 4))
    fake_returns = [(100, 'increase'), (-100, 'decrease')]
    print('Testing policy updates...')
    for action in range(NUM_ACTIONS):
        actions = action * np.ones(batch_size)
        for fake_return, expected in fake_returns:
            # Reinitialize session because ADAM optimizer builds momentum
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                # Compare new and old probabilities
                old_probs = policy.predict(sess, state)
                policy.update(sess, state, actions, [fake_return])
                new_probs = policy.predict(sess, state)
                print('Action {} probability should {}:'.format(action, expected), end=' ')
                print(np.mean(new_probs - old_probs, axis=0))

    print('\nTesting baseline updates...')
    targets = 100 * np.ones(batch_size)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        old_value = baseline.predict(sess, state)
        baseline.update(sess, state, targets)
        new_value = baseline.predict(sess, state)
        value_change = np.mean(new_value - old_value)
        print('Value of states should increase: {}'.format(value_change))


def summary_writer_op(sess, logdir, policy_net, value_net):
    '''
    Merge all summaries and returns an function to
    write the summary
    '''
    # Check if path already exists
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = tf.summary.FileWriter(logdir, sess.graph)

    # Create placeholders to track some statistics
    episode_reward = tf.placeholder(name='episode_reward',
                                         shape=(),
                                         dtype=tf.float32)
    episode_length = tf.placeholder(name='episode_length',
                                         shape=(),
                                         dtype=tf.float32)
    # Create some summaries
    tf.summary.scalar('reward', episode_reward)
    tf.summary.scalar('episode_length', episode_length)

    # Merge all summaries
    merged = tf.summary.merge_all()
    global_step = slim.get_or_create_global_step()

    def summary_writer(states, actions, returns, targets, reward, length):
        feed_dict = {
            policy_net.states: states,
            policy_net.actions: actions,
            policy_net.returns: returns,
            value_net.states: states,
            value_net.targets: targets,
            episode_reward: reward,
            episode_length: length
        }
        summary, step = sess.run([merged, global_step],
                                 feed_dict=feed_dict)
        # Write summary
        writer.add_summary(summary, step)

    return summary_writer


if __name__ == '__main__':
    import numpy as np
    NUM_ACTIONS = 2
    # Test if updates are working as expected
    test_updates(learning_rate=3e-4, num_actions=NUM_ACTIONS)
