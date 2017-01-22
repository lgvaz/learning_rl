import numpy as np


class ReplayMemory:
    '''
    Creates a buffer that stores experience (replays)

    Args:
        max_replays: The maximum number of replays
        num_features: Number of possible observations on gym env
    '''
    def __init__(self, max_replays, num_features):
        self.max_replays = max_replays
        self.num_features = num_features
        # Allocate memory
        self.states = np.empty((max_replays, self.num_features), dtype=np.float32)
        self.actions = np.empty(max_replays, dtype=np.int32)
        self.rewards = np.empty(max_replays, dtype=np.float32)
        self.done = np.empty(max_replays, dtype=np.bool)
        # Create "pointers"
        self.size = 0
        self.current = 0

    def add_replay(self, state, action, reward, done):
        '''  Add experience to memory '''
        self.states[self.current] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.done[self.current] = done
        # Update memory actual size
        if self.size != self.max_replays:
            self.size += 1
        self.current = (1 + self.current) % self.max_replays

    def sample(self, batch_size):
        '''
        '''
        batch_states = np.empty((batch_size, self.num_features), dtype=np.float32)
        batch_next_states = np.empty((batch_size, self.num_features), dtype=np.float32)
        batch_actions = np.empty(batch_size, dtype=np.int32)
        batch_rewards = np.empty(batch_size, dtype=np.float32)
        batch_done = np.empty(batch_size, dtype=np.bool)

        for i_batch in range(batch_size):
            idx = np.random.randint(0, self.size - 1)
            batch_states[i_batch] = self.states[idx]
            batch_next_states[i_batch] = self.states[idx + 1]
            batch_actions[i_batch] = self.actions[idx]
            batch_rewards[i_batch] = self.rewards[idx]
            batch_done[i_batch] = self.done[idx]

        return (batch_states,
                batch_next_states,
                batch_actions,
                batch_rewards,
                batch_done)


if __name__ == '__main__':
    # Testing ReplayMemory
    num_features = 4
    max_replays = 100
    test_mem = ReplayMemory(max_replays, num_features)
    for i_replay in range(4 * max_replays):
        state = [i_replay] * num_features
        action = np.random.choice(np.arange(3))
        reward = 0
        done = np.random.choice(np.arange(2), p=[0.8, 0.2])
        test_mem.add_replay(state, action, reward, done)
    print('Mem size: {}'.format(test_mem.size))
    print('Replays of first indexes: \n')
    print(test_mem.states[:5])
    b_states, b_next_states, b_actions, b_rewards, b_dones = test_mem.sample(32)
    print('Batch states shape: {}'.format(b_states.shape))
    print('Batch next states shape: {}'.format(b_next_states.shape))
    print('Batch actions shape: {}'.format(b_actions.shape))
    print('Batch rewards shape: {}'.format(b_rewards.shape))
    print('Batch dones shape: {}'.format(b_dones.shape))
