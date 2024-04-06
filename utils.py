import numpy as np
import matplotlib.pyplot as plt

def visualize_pricing_strategy(history):
    # Splitting the history into components for plotting
    prices, market_demands, captured_demands, revenues = zip(*history)
    bottoms = [md - cd for md, cd in zip(market_demands, captured_demands)]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(range(len(history)), market_demands, label='Market Demand', color='lightblue', alpha=0.6)
    ax1.bar(range(len(history)), captured_demands, bottom=bottoms, label='Captured Demand', color='blue', alpha=0.6)
    ax2.plot(range(len(history)), prices, 'ro-', label='Price')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Demand', color='blue')
    ax2.set_ylabel('Price', color='red')
    ax1.set_title('Market and Captured Demand with Prices')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.legend(loc='upper left')
    ax2.legend(['Price'], loc='upper right')
    plt.figure()
    plt.plot(range(len(history)), revenues, 'g^-', label='Revenue')
    plt.xlabel('Step')
    plt.ylabel('Revenue')
    plt.title('Revenue Over Time')
    plt.legend()
    plt.show()
    plt.show()
