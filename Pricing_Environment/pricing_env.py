import gymnasium
from gymnasium import spaces
from Pricing_Environment.demand_calculator import DemandCalculator
from Pricing_Environment.action_strategy import ActionStrategy
import numpy as np

class PricingEnvironment(gymnasium.Env):
    metadata = {'render_modes': ['text', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, product_config, demand_calculator_config, action_strategy_config,render_mode=None, ):
        super().__init__()
        self.min_price = product_config["min_price"]
        self.max_price = product_config["max_price"]
        self.current_demand = product_config["initial_demand"]
        self.current_price = np.mean([self.min_price, self.max_price])
        
        # Initialize the demand calculator with its config
        self.demand_calculator = DemandCalculator(demand_calculator_config["price_probability_ranges"])
        
        # Initialize the action strategy with its config
        self.action_strategy = ActionStrategy(action_strategy_config["action_probabilities"])
        
        self.action_space = spaces.Discrete(len(action_strategy_config["action_probabilities"]))
        self.observation_space = spaces.Box(low=np.array([0, self.min_price]), high=np.array([1, self.max_price]), dtype=np.float32)
        
        self.render_mode = render_mode
        self.history = []

    def step(self, action, external_market_demand):
        # Map chosen action to price change
        price_change_map = {
            0: -10,  # Decrease significantly
            1: -5,   # Decrease slightly
            2: 0,    # Keep price
            3: 5,    # Increase slightly
            4: 10    # Increase significantly
        }

        price_change = price_change_map[action]
        self.current_price = np.clip(self.current_price + price_change, self.min_price, self.max_price)

        # Update current demand based on external market demand
        self.market_demand = external_market_demand
        self.current_demand = self.demand_calculator.calculate_demand(self.current_price, external_market_demand)

        reward = self.current_demand * self.current_price
        state = np.array([self.current_demand, self.current_price])
        done = False

        revenue = self.current_price * self.current_demand
        
        # Record with revenue
        self.history.append((self.current_price, self.market_demand, self.current_demand, revenue))
    

        return state, reward, done, {}

    def reset(self):
        self.market_demand = np.random.uniform(0, 1)
        self.current_price = np.mean([self.min_price, self.max_price])
        self.history = []  # Reset history
        return np.array([self.market_demand, self.current_price])

    def render(self, mode='text'):
        if mode == 'text':
            latest_record = self.history[-1] if self.history else (self.current_price, self.market_demand, self.current_demand)
            print(f"Price: {latest_record[0]}, Market Demand: {latest_record[1]}, Current Demand: {latest_record[2]}")
    
    def close(self):
        pass
