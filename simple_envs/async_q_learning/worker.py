import time
import gym
from gym import wrappers
import itertools
import numpy as np
import tensorflow as tf
from threading import Thread, Lock
from estimators import *

class Worker:
    def __init__(self, env_name, num_actions, num_workers, num_steps, stop_exploration, discount_factor, online_update_step, target_update_step, online_net, target_net, sess, coordinator):
        self.online_net = online_net
        self.target_net = target_net
        self.sess = sess
#    def __init__(self, env_name, num_actions, num_workers, num_steps, coordinator):
        self.env_name = env_name
        self.num_actions = num_actions
        self.num_workers = num_workers
        self.num_steps = num_steps
        self.stop_exploration = stop_exploration
        self.discount_factor = discount_factor
        self.online_update_step = online_update_step
        self.target_update_step = target_update_step
        self.coord = coordinator
        # Target update operation
        self.target_update = target_update = copy_vars(sess=sess, from_scope='online', to_scope='target')
        # Shared global step
        self.global_step = 0
        # Define final epsilon
        self.final_epsilon = np.array([0.1, 0.01, 0.5])
        # Creates locks
        self.global_step_lock = Lock()
        self.create_env_lock = Lock()
        self.render_lock = Lock()

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
            env = gym.make(self.env_name)

        # TODO: add monitor
        # Create a monitor for only one env
        #     env = wrappers.Monitor(env=env, directory='test', force=True)

        get_epsilon = get_epsilon_op(self.final_epsilon, self.stop_exploration)
        # Repeat until coord requests a stop
        while not self.coord.should_stop():
            state = env.reset()
            experience = []
            ep_reward = 0
            for local_step in itertools.count():
                # with self.render_lock:
                #     env.render()
                # Take random action
#                action = np.random.choice(np.arange(self.num_actions))
                # Increment global step
                with self.global_step_lock:
                    self.global_step += 1
                # Compute action values with online net
                action_values = self.online_net.predict(self.sess, state[np.newaxis])
                # Compute epsilon and choose an action based on a egreedy policy
                epsilon = get_epsilon(self.global_step)
                action_probs = egreedy_policy(action_values, epsilon)
                action = np.random.choice(np.arange(self.num_actions), p=action_probs)
                # Do the action
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward

                # Calculate TD target
                next_action_values = self.target_net.predict(self.sess, next_state[np.newaxis])
                next_max_action_value = np.max(next_action_values)
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
                with self.global_step_lock:
                    if self.global_step % self.target_update_step == 0:
                        print('Step {}, updating target network'.format(self.global_step))
                        self.target_update()

                # If the maximum step is reached, request all threads to stop
                if self.global_step == self.num_steps:
                    # with self.global_step_lock:
                        self.coord.request_stop()

                # Update state
                if done:
                    break
                state = next_state

            print('Step: {} | Reward: {}'.format(self.global_step, ep_reward))


if __name__ == '__main__':
    coord = tf.train.Coordinator()
    workers = Worker('MountainCar-v0', 2, 2, 4000, coord)
    workers.run()

    # Wait until all threads are done
    print('Done!')
