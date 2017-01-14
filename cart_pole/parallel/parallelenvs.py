import gym
import itertools
import numpy as np
from threading import Thread, Lock
from queue import Queue


def reward_return(rewards, discount_factor=0.99):
    ''' Calculate the return from step t '''
    G = []
    G_sum = 0
    # Start calculating from the last reward
    for t, reward in enumerate(reversed(rewards)):
        G_sum = G_sum * discount_factor + reward
        G.insert(0, G_sum)
    # Normalize returns
    G = (G - np.mean(G)) / np.std(G)
    return G


class ParallelEnvs:
    '''
    Runs parallel copies of a gym env

    Args:
        env_name: The name of the gym env to be used
    '''
    def __init__(self, env_name, num_actions, policy_net, sess, render=False):
        self.env_name = env_name
        self.num_actions = num_actions
        self.policy_net = policy_net
        self.sess = sess
        self.render = render
        # Create a queue that assign jobs
        self.q = Queue()
        # Create locks to prevent workers to modify
        # the same variable at the same time
        self.experience_lock = Lock()
        self.graph_lock = Lock()
        self.create_env_lock = Lock()
        self.render_lock = Lock()

    def create_workers(self, num_workers):
        # Create workers
        for i_worker in range(num_workers):
            t = Thread(target=self._worker, args=(i_worker,))
            t.daemon = True
            t.start()

    def run(self, batch_size):
        ''' Generates "batch size" episodes '''
        # Create shared experience list
        with self.experience_lock:
            self.shared_experience = []
        # Put jobs on the queue
        for i_episode in range(batch_size):
            self.q.put(i_episode)
        # Wait for all jobs to be completed
        self.q.join()
        return self.shared_experience

    def _worker(self, name):
        ''' Creates a parallel gym env '''
        print('Starting worker {}'.format(name))
        with self.create_env_lock:
            env = gym.make(self.env_name)
        # Keep looking for jobs on queue
        while True:
            work = self.q.get()
            self._run_episode(env)
            self.q.task_done()

    def _run_episode(self, env):
        states = []
        actions = []
        rewards = []
        state = env.reset()
        # Repeat until episode is finished
        for i_step in itertools.count():
            if self.render:
                with self.render_lock:
                    env.render()
            # Choose an action
            with self.graph_lock:
                action_probs = self.policy_net.predict(self.sess, state[np.newaxis])
                action = np.random.choice(np.arange(self.num_actions), p=np.squeeze(action_probs))
            # Do the action
            next_state, reward, done, _ = env.step(action)
            # Store the results
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if done:
                # Calculate returns
                returns = reward_return(rewards)
                experience = zip(states, actions, rewards, returns)
                # Write experience
                with self.experience_lock:
                    self.shared_experience.extend(experience)
                break
            state = next_state


if __name__ == '__main__':
    envs = ParallelEnvs(env_name='CartPole-v0', render=True)
    envs.create_workers(8)
    experience = envs.run(32)
    print(experience[0])
    print(len(experience))
