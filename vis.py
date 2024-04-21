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
