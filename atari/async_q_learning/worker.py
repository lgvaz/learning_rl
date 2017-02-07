import gym
import itertools
import numpy as np
import tensorflow as tf
from threading import Thread, Lock
from estimators import *
from atari_envs import AtariWrapper

class Worker:
    def __init__(self, env_name, num_actions, num_workers, num_steps,
                 stop_exploration, final_epsilon, discount_factor,
                 online_update_step, target_update_step, online_net, target_net, global_step,
                 double_learning, sess, coord, saver, summary_writer, savepath, videodir):
        self.env_name = env_name
        self.num_actions = num_actions
        self.num_workers = num_workers
        self.num_steps = num_steps
        self.stop_exploration = stop_exploration
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.online_update_step = online_update_step
        self.target_update_step = target_update_step
        self.online_net = online_net
        self.target_net = target_net
        self.global_step = global_step
        self.double_learning = double_learning
        self.num_stacked_frames = 4
        self.sess = sess
        self.coord = coord
        self.saver = saver
        self.summary_writer = summary_writer
        self.savepath = savepath
        self.videodir = videodir
        # Target update operation
        self.target_update = copy_vars(sess=sess, from_scope='online', to_scope='target')
        self.target_update()
        # Shared global step
        #self.global_step = slim.get_or_create_global_step()
        self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)
        # Creates locks
        self.global_step_lock = Lock()
        self.create_env_lock = Lock()
        self.render_lock = Lock()
        self.summary_lock = Lock()

    def run(self):
        # Create workers
        threads = []
        for i_worker in range(self.num_workers):
            t = Thread(target=self._run_worker, args=(i_worker,))
            threads.append(t)
            t.daemon = True
            t.start()
        self.coord.join(threads)

    def _run_worker(self, name):
        '''
        Creates a parallel thread that runs a gym environment
        and updates the networks
        '''
        print('Starting worker {}...'.format(name))
        # Starting more than one env at once may break gym
        with self.create_env_lock:
            # Record videos of only one env
            if name == 0:
                env = AtariWrapper(self.env_name, self.videodir)
            else:
                env = AtariWrapper(self.env_name)
            # Configure the maximum number of steps
            # env.spec.tags.update({'wrapper_config.TimeLimit.max_episode_steps': 1500})

        get_epsilon = get_epsilon_op(self.final_epsilon, self.stop_exploration)
        # Repeat until coord requests a stop
        while not self.coord.should_stop():
            state = env.reset()
            # Create first stacked frames
            experience = []
            ep_reward = 0
            for local_step in itertools.count():
                # with self.render_lock:
                #     env.render()
                # Increment global step
                with self.global_step_lock:
                    self.sess.run(self.increment_global_step)
                    global_step_value = self.sess.run(self.global_step)
                # Compute action values with online net
                action_values = self.online_net.predict(self.sess, state[np.newaxis])
                # Compute epsilon and choose an action based on a egreedy policy
                epsilon = get_epsilon(global_step_value)
                action_probs = egreedy_policy(action_values, epsilon)
                action = np.random.choice(np.arange(self.num_actions), p=action_probs)
                # Do the action
                next_state, reward, done, _ = env.step(action)
                # Build frames history
                ep_reward += reward

                # Calculate simple Q learning max action value
                if self.double_learning == 'N':
                    next_action_values = self.target_net.predict(self.sess, next_state[np.newaxis])
                    next_max_action_value = np.max(next_action_values)
                # Calculate double Q learning max action value
                if self.double_learning == 'Y':
                    next_action_values = self.online_net.predict(self.sess, next_state[np.newaxis])
                    next_action = np.argmax(next_action_values)
                    next_action_values_target = self.online_net.predict(self.sess, next_state[np.newaxis])
                    next_max_action_value = np.squeeze(next_action_values_target)[next_action]
                # Calculate TD target
                td_target = reward + (1 - done) * self.discount_factor * next_max_action_value
                # Store experience
                experience.append((state, action, td_target))

                # Update online network
                if local_step % self.online_update_step == 0 or done:
                    # Unpack experience
                    states, actions, targets = zip(*experience)
                    experience = []
                    # Updating using Hogwild! (without locks)
                    self.online_net.update(self.sess, states, actions, targets)

                # Update target network
                if global_step_value % self.target_update_step == 0:
                    print('Updating target network...')
                    self.target_update()

                # Write logs and checkpoint
                if global_step_value % 10000 == 0:
                    print('Writing summary...')
                    self.summary_writer(states, actions, targets, ep_reward, local_step, global_step_value)
                    print('Saving model...')
                    self.saver.save(self.sess, self.savepath)

                # If the maximum step is reached, request all threads to stop
                if global_step_value >= self.num_steps:
                    # with self.global_step_lock:
                        self.coord.request_stop()

                # Update state
                if done:
                    break
                state = next_state

            print('Step: {} | Reward: {} | Length {}'.format(global_step_value, ep_reward, local_step))
