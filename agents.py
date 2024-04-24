import numpy as np

# Definition of the Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_rate=0.95,
                 exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(range(self.action_size))
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        future_value = np.max(self.q_table[next_state]) if not done else 0
        new_value = old_value + self.learning_rate * (reward + self.discount_rate * future_value - old_value)
        self.q_table[state, action] = new_value


class UCB:
    def __init__(self, n_actions, exploration_coefficient=2):
        self.n_actions = n_actions
        self.action_counts = np.zeros(n_actions, dtype=int)
        self.total_rewards = np.zeros(n_actions, dtype=float)
        self.exploration_coefficient = exploration_coefficient
    
    def select_action(self):
        for action in range(self.n_actions):
            if self.action_counts[action] == 0:
                return action
        total_counts = sum(self.action_counts)
        ucb_values = self.total_rewards / self.action_counts + self.exploration_coefficient * np.sqrt(np.log(total_counts) / self.action_counts)

        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.total_rewards[action] += reward
        
class MarkovDecisionProcess:
    def __init__(self, actions, epsilon=0.1):
        """
        Initialize with a dictionary of action probabilities and an exploration factor.
        """
        self.actions = actions  # Dictionary of action probabilities
        self.epsilon = epsilon  # Exploration factor
        self.total_rewards = {action: 0.0 for action in actions}
        self.action_counts = {action: 0 for action in actions}

    def choose_action(self):
        """
        Chooses an action based on epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(list(self.actions.keys()))
        else:
            # Exploitation: choose based on modified probabilities
            actions, probabilities = zip(*self.actions.items())
            probabilities = np.array(probabilities)
            if np.any(probabilities < 0) or not np.isclose(np.sum(probabilities), 1):
                # Normalize if there are any issues
                probabilities = np.maximum(probabilities, 0)  # Eliminate negative probabilities
                probabilities /= np.sum(probabilities)  # Normalize to sum to 1
            return np.random.choice(actions, p=probabilities)

    def update_probabilities(self, action, reward):
        """
        Updates the probabilities of actions based on the received reward.
        """
        self.total_rewards[action] += reward
        self.action_counts[action] += 1

        total_rewards = sum(self.total_rewards.values())
        for act in self.actions:
            if total_rewards > 0:
                self.actions[act] = max(self.total_rewards[act] / total_rewards, 0)  # Ensure non-negative
            else:
                n = len(self.actions)
                self.actions[act] = 1.0 / n  # Reset to uniform if total rewards are zero

        # Ensure probabilities sum to 1
        total_prob = sum(self.actions.values())
        for act in self.actions:
            self.actions[act] /= total_prob


class SARSA:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(range(self.action_size))
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        old_value = self.q_table[state, action]
        future_value = self.q_table[next_state][next_action] if not done else 0
        new_value = old_value + self.learning_rate * (reward + self.discount_rate * future_value - old_value)
        self.q_table[state, action] = new_value


def encode_state(market_demand, boundaries):
    """
    Encodes the market demand into a discrete state based on specified boundaries.

    Args:
        market_demand (float): The current market demand from the environment's state.
        boundaries (list of floats): The boundaries between different demand bins. Should be sorted in ascending order.

    Returns:
        int: A discrete integer representing the encoded state. Returns the index of the bin into which the market demand falls.
    """
    # Iterate over the boundaries to determine the correct bin
    for i, boundary in enumerate(boundaries):
        if market_demand < boundary:
            return i
    # If the demand is greater than all boundaries, return the last bin index
    return len(boundaries)


