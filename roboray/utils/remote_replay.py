import random
import ray

@ray.remote
class RemoteReplayBuffer(object):
	"""
	This class represent the remote replay buffer used by 
	distributed algorithms.
	"""
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size) 
        self.max_size = max_size
        
    def store_transition(self, state, action, reward, state_, done):
        """
        This function stores a transition tuple to the buffer.
        
        Args:
        
        Returns:
        
        Raises:
        """
        experience = (state, action, np.array([reward]), state_, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        This function randomly samples a batch from the buffer.
        
        Args:
            batch_size: the size of the batch to be sampled 
            
        Returns:
            Lists of states, actions, rewards, next_states and done flag.
            
        Raises:
            <No raises implemented>
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer) 