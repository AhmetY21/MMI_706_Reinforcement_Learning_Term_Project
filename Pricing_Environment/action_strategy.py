import numpy as np

class ActionStrategy:
    def __init__(self, actions, epsilon=0.1):
        """
        Initialize the ActionStrategy with actions and an epsilon for exploration.
        :param actions: A dictionary of action probabilities.
        :param epsilon: The probability of choosing a random action.
        """
        self.actions = actions
        self.epsilon = epsilon
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
            # Exploitation: choose the best action based on current probabilities
            actions, probabilities = zip(*self.actions.items())
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
                self.actions[act] = self.total_rewards[act] / total_rewards
            else:
                n = len(self.actions)
                self.actions[act] = 1.0 / n  # Reset to uniform if total rewards are zero

        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(self.actions.values())
        for act in self.actions:
            self.actions[act] /= total_prob