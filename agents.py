import numpy as np
import torch

# Check if CUDA (GPU support) is available and choose accordingly
device = torch.device("cpu")


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

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
    def __init__(self,price_change_map, exploration_coefficient=2):
        self.n_actions = len(price_change_map)
        self.action_counts = np.zeros(len(price_change_map), dtype=int)
        self.total_rewards = np.zeros(len(price_change_map), dtype=float)
        self.exploration_coefficient = exploration_coefficient
        self.cumulative_price_change = 0  # Initialize cumulative price change
        # Map of action indices to price changes
        self.price_change_map = price_change_map

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
        # Update cumulative price change based on the action taken
        self.cumulative_price_change += self.price_change_for_action(action)

    def price_change_for_action(self, action):
        # Retrieve the price change associated with the action
        return self.price_change_map[action]

    def get_cumulative_price_change(self):
        return self.cumulative_price_change


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




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)  # Standard deviation must be positive
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(z) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.network(sa)

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.discount = 0.99
        self.tau = 0.005
        self.policy_delay = 2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.actor.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.critic1_target.to(self.device)
        self.critic2_target.to(self.device)

    def train(self, replay_buffer, batch_size=256):
        for it in range(batch_size):
            # Sample a batch of transitions from the replay buffer
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).to(self.device)

            # Compute the target Q value
            with torch.no_grad():
                next_action = self.actor.sample(next_state)
                target_Q1 = self.critic1_target(next_state, next_action)
                target_Q2 = self.critic2_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1 - done) * self.discount * target_Q)

            # Get current Q estimates
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()

            # Delayed policy updates
            if it % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic1(state, self.actor.sample(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update the target networks
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




        
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

