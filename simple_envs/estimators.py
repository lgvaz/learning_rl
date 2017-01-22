import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class QNet:
    '''
    Neural network for approximating Q values
    '''

    def __init__(self, num_features, num_actions, learning_rate, scope, clip_grads, create_summary=False):
        '''
        Creates the network

        Args:
            num_features: Number of possible observations of environment
            num_actions: Number of possible actions
            learning_rate: Learning rate used when performing gradient descent
            scope: The scope used by the network
            clip_grads: Whether gradients should be clipped to a range
                        of [-1, 1] or not
            create_summary: Whether the network should create summaries or not,
                            only the main network should create summaries
        '''
        # Placeholders
        self.states = tf.placeholder(name='states',
                                     shape=[None, num_features],
                                     dtype=tf.float32)
        self.targets = tf.placeholder(name='td_targets',
                                      shape=[None],
                                      dtype=tf.float32)
        self.actions = tf.placeholder(name='chosen_actions',
                                       shape=[None],
                                       dtype=tf.int32)

        # Define network architecture
        with tf.variable_scope(scope):
            self.fc = slim.fully_connected(self.states, 16)
            self.fc = slim.fully_connected(self.fc, 16)
            self.outputs = slim.fully_connected(self.fc, num_actions,
                                                activation_fn=None)

        # Optimization process
        batch_size = tf.shape(self.states)[0]
        # Pick only the actions which were chosen
        # action_ids = (i_batch * NUM_ACTIONS) + action
        actions_ids = tf.range(batch_size) * num_actions + self.actions
        actions_value = tf.gather(tf.reshape(self.outputs, [-1]), actions_ids)
        # Calculate mean squared error
        self.loss = tf.reduce_mean(tf.squared_difference(self.targets, actions_value))
        opt = tf.train.AdamOptimizer(learning_rate)
        # Get list of variables given by scope
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        # Calcuate gradients
        grads_and_vars = opt.compute_gradients(self.loss, local_vars)
        if clip_grads == True:
            grads_and_vars = [(tf.clip_by_value(grad, -1, 1), var)
                             for grad, var in grads_and_vars]
        self.global_step = slim.get_or_create_global_step()
        self.training_op = opt.apply_gradients(grads_and_vars, self.global_step)

        if create_summary:
            # Add summaries
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('max_q', tf.reduce_max(self.outputs))
            tf.summary.scalar('average_q', tf.reduce_mean(self.outputs))
            tf.summary.histogram('q_values', self.outputs)

    def predict(self, sess, states):
        '''
        Compute the value of each action for each state

        Args:
            sess: Tensorflow session to be used
            states: Environment observations
        '''
        return sess.run(self.outputs, feed_dict={self.states: states})

    def update(self, sess, states, actions, targets):
        '''
        Performs the optimization process

        Args:
            sess: Tensorflow session to be used
            states: Enviroment observations
            actions: Actions to be updated (Probally performed actions)
            targets: TD targets
        '''
        feed_dict = {self.states: states,
                     self.actions: actions,
                     self.targets: targets}
        sess.run(self.training_op, feed_dict=feed_dict)

    def create_summary_op(self, sess, logdir):
        '''
        Creates summary writer

        Args:
            sess: Tensorflow session to be used
            logdir: Directory for saving summary

        Returns:
            A summary writer operation
        '''
        # Check if path already exists
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Create placeholders to track some statistics
        episode_reward = tf.placeholder(name='episode_reward',
                                             shape=[],
                                             dtype=tf.float32)
        episode_length = tf.placeholder(name='episode_length',
                                             shape=[],
                                             dtype=tf.float32)
        epsilon = tf.placeholder(name='epsilon',
                                 shape=[],
                                 dtype=tf.float32)
        # Create some summaries
        tf.summary.scalar('reward', episode_reward)
        tf.summary.scalar('episode_length', episode_length)
        tf.summary.scalar('epsilon', epsilon)
        # Merge all summaries
        merged = tf.summary.merge_all()
        global_step = slim.get_or_create_global_step()

        def summary_writer(states, actions, targets, reward, length, epsilon):
            feed_dict = {
                self.states: states,
                self.actions: actions,
                self.targets: targets,
                episode_reward: reward,
                episode_length: length,
                epsilon: epsilon
            }
            summary, step = sess.run([merged, global_step],
                                     feed_dict=feed_dict)

            # Write summary
            writer.add_summary(summary, step)

        return summary_writer


def copy_vars(sess, from_scope, to_scope):
    '''
    Create operations to copy variables (weights) between two graphs

    Args:
        sess: The current tensorflow session
        from_scope: name of graph to copy varibles from
        to_scope: name of graph to copy varibles to
    '''
    # Get variables within defined scope
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    # Create operations that copy the variables
    op_holder = [to_var.assign(from_var) for from_var, to_var in zip(from_vars, to_vars)]

    def run_op():
        # Runs the operation
        sess.run(op_holder)

    return run_op


def egreedy_policy(q_value, epsilon):
    '''
    Returns actions probabilities based on an epsilon greedy policy
    '''
    num_actions = len(np.squeeze(q_value))
    actions = (np.ones(num_actions) * epsilon) / num_actions
    best_action = np.argmax(np.squeeze(q_value))
    actions[best_action] += 1 - epsilon
    return actions
