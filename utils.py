import numpy as np
import matplotlib.pyplot as plt


def visualise_pricing_strategy(all_histories):
    # Flatten all episode data into a single list for each category
    prices, market_demands, captured_demands, revenues = zip(*[item for episode in all_histories for item in episode])

    # Calculate uncaptured demands
    uncaptured_demands = [md - cd for md, cd in zip(market_demands, captured_demands)]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(range(len(prices)), captured_demands, label='Captured Demand', color='blue', alpha=0.6)
    ax1.bar(range(len(prices)), uncaptured_demands, bottom=captured_demands, label='Uncaptured Demand', color='lightblue', alpha=0.6)
    ax2.plot(range(len(prices)), prices, 'ro-', label='Prisce')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Demand', color='blue')
    ax2.set_ylabel('Price', color='red')
    ax1.set_title('Market and Captured Demand with Prices')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.legend(loc='upper left')
    ax2.legend(['Price'], loc='upper right')

    plt.show()


def visualise_demand_data(demand_data):
    """
    Visualizes demand data for a single episode using a bar chart.

    Args:
        demand_data (array): Demand data for an episode.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(demand_data)), demand_data, color='skyblue')
    plt.title('Demand Data for Single Episode')
    plt.xlabel('Step')
    plt.ylabel('Demand')
    plt.show()


def visualise_pricing_strategy(histories):
    # Assume 'histories' is a list of history lists, one per episode
    all_prices = []
    all_market_demands = []
    all_captured_demands = []
    all_revenues = []

    for history in histories:
        prices, market_demands, captured_demands, revenues = zip(*history)
        all_prices.append(prices)
        all_market_demands.append(market_demands)
        all_captured_demands.append(captured_demands)
        all_revenues.append(revenues)

    # Average captured demands per episode
    average_captured_demands = [np.mean(cd) for cd in all_captured_demands]

    # Visualization of Prices and Demands
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Average prices over episodes
    average_prices = [np.mean(prices) for prices in all_prices]
    steps = list(range(len(histories)))

    # Plot average captured demands per episode
    ax1.plot(steps, average_captured_demands, label='Average Captured Demand', color='blue', marker='o', linestyle='--')
    ax2.plot(steps, average_prices, 'ro-', label='Average Price')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Captured Demand', color='blue')
    ax2.set_ylabel('Average Price', color='red')
    ax1.set_title('Average Captured Demand and Prices Per Episode')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

    # Separate figure for average revenue over time
    average_revenues = [np.mean(revenues) for revenues in all_revenues]
    plt.figure()
    plt.plot(steps, average_revenues, 'g^-', label='Average Revenue Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Revenue')
    plt.title('Average Revenue Over Episodes')
    plt.legend()
    plt.show()

def visualise_episode_rewards(episode_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Reward per Episode', marker='o', linestyle='-')
    plt.title('Reward Received Over Each Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, which='both', linestyle='-')
    plt.legend()
    plt.show()

def external_demand_function(min=0.5,max=1.5,dist='uniform'):
    return np.random.uniform(0.5, 1.5)

def visualise_total_rewards_ucb(episode_rewards):
    """
    Visualizes the total rewards received in each episode.

    Args:
        episode_rewards (list): List of total rewards per episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, marker='o', linestyle='-', color='b')
    plt.title('Total Rewards Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

def visualise_rewards_and_prices(episode_rewards, offered_prices, rewards_title='Total Rewards Per Episode', prices_title='Average Offered Prices Per Episode'):
    """
    Visualizes the rewards and offered prices received in each episode on the same plot with dual y-axes.

    Args:
        episode_rewards (list): List of total rewards per episode.
        offered_prices (list): List of offered prices for each episode.
        rewards_title (str): Title for the rewards plot.
        prices_title (str): Title for the offered prices plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plotting the rewards on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(rewards_title, color=color)
    ax1.plot(episode_rewards, marker='o', linestyle='-', color=color, label=rewards_title + ' (left axis)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Creating a second y-axis for the offered prices
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel(prices_title, color=color)  
    ax2.plot(offered_prices, marker='x', linestyle='--', color=color, label=prices_title + ' (right axis)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Adding a title and legend
    plt.title('Episode Rewards and Offered Prices')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    
    plt.show()



def visualise_rewards(episode_rewards, title='Total Rewards Per Episode'):
    """
    Visualizes the rewards received in each episode.

    Args:
        episode_rewards (list): List of total rewards per episode.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Reward per Episode')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    plt.show()

def visualise_avg_offered_price(offered_prices, title='Average Offered Prices For Each Episode'):
    """
    Visualizes the rewards received in each episode.

    Args:
        offered_prices (list): List of offered prices of each episode.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(offered_prices, marker='o', linestyle='-', color='b', label='Offered Price of Each Episode')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Average Offered Prices')
    plt.grid(True)
    plt.legend()
    plt.show()


def visualise_ucb_confidence(ucb_agent):
    """
    Visualizes the UCB confidence levels for each action.

    Args:
        ucb_agent (UCB): The UCB agent instance after simulation.
    """
    # Calculate the UCB values one final time
    if sum(ucb_agent.action_counts) == 0:
        return  # Avoid division by zero if no actions were taken
    ucb_values = ucb_agent.total_rewards / ucb_agent.action_counts + \
                 np.sqrt(2 * np.log(sum(ucb_agent.action_counts)) / ucb_agent.action_counts)
    
    plt.figure(figsize=(12, 6))
    actions = range(len(ucb_values))
    plt.bar(actions, ucb_values, color='green')
    plt.title('UCB Confidence Levels for Each Action')
    plt.xlabel('Actions')
    plt.ylabel('UCB Value')
    plt.xticks(actions)  # Ensure that each action is labeled with its index
    plt.grid(True)
    plt.show()




def visualise_ucb_demand_rewards(num_episodes, total_market_demands, total_captured_demands, episode_rewards):
    """
    Visualizes total market demands, captured demands, and rewards per episode using a stacked bar plot and a line plot overlay.

    Args:
        num_episodes (int): Number of episodes.
        total_market_demands (list or np.array): Total market demands per episode.
        total_captured_demands (list or np.array): Total captured demands per episode.
        episode_rewards (list or np.array): Total rewards per episode.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    indices = np.arange(num_episodes)  # the label locations

    # Convert lists to numpy arrays for element-wise operations
    total_market_demands = np.array(total_market_demands)
    total_captured_demands = np.array(total_captured_demands)

    # Captured demands at the base
    ax1.bar(indices, total_captured_demands, label='Total Captured Demand', color='blue', alpha=0.7)

    # Remaining market demands calculated as the difference and plotted above captured demands
    remaining_market_demands = total_market_demands - total_captured_demands
    ax1.bar(indices, remaining_market_demands, label='Remaining Market Demand', color='gray', alpha=0.7, bottom=total_captured_demands)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Demand')
    ax1.set_title('Market and Captured Demand per Episode with Total Rewards')
    ax1.legend(loc='upper left')

    # Create a twin axis for the rewards line plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('Rewards')
    ax2.plot(indices, episode_rewards, 'r-', label='Total Rewards')
    ax2.legend(loc='upper right')

    plt.show()



def visualise_sac_test_rewards(test_rewards):
    """
    Visualizes the test rewards of the SAC agent over a series of episodes.

    Args:
    test_rewards (list of float): A list containing the total rewards obtained by the SAC agent in each test episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test_rewards, marker='o', linestyle='-', color='blue')
    plt.title('SAC Test Rewards Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()
