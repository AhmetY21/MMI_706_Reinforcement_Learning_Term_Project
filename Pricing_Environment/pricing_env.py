import gymnasium
from gymnasium import spaces
from Pricing_Environment.demand_calculator import DemandDataGenerator,DemandCalculator
from Pricing_Environment.action_strategy import ActionStrategy
import numpy as np

class PricingEnvironment(gymnasium.Env):
    metadata = {'render_modes': ['text', 'rgb_array'], 'render_fps': 4}

    
    def __init__(self, product_config, demand_calculator_config, action_strategy_config, is_continuous=False, render_mode=None):
        super().__init__()
        self.min_price = product_config["min_price"]
        self.max_price = product_config["max_price"]
        self.current_demand = product_config["initial_demand"]
        self.current_price = np.mean([self.min_price, self.max_price])
        self.demand_generator = DemandDataGenerator(demand_calculator_config['low'],demand_calculator_config['high'],demand_calculator_config['seasonality'])  # Configure as needed

        # Initialize the demand calculator with its config
        self.demand_calculator = DemandCalculator(demand_calculator_config["price_probability_ranges"])
        
        # Initialize the action strategy with its config
        self.action_strategy = ActionStrategy(action_strategy_config["action_probabilities"])
        self.price_change_map = action_strategy_config.get("price_change_map", {})
        
        # Action space definition based on whether actions are continuous or discrete
        self.is_continuous = is_continuous
        if is_continuous:
            self.action_space = spaces.Box(low=np.array([self.min_price]), high=np.array([self.max_price]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(action_strategy_config["action_probabilities"]))
        
        self.observation_space = spaces.Box(low=np.array([0, self.min_price]), high=np.array([1, self.max_price]), dtype=np.float32)
        
        self.render_mode = render_mode
        self.history = []

    def generate_market_demand(self):
        # Generate market demand using the DemandDataGenerator
        return self.demand_generator.generate()  # Directly use the generated array


    def step(self, action, external_market_demand):
        if not self.is_continuous:
            if action in self.price_change_map:
                price_change = self.price_change_map[action]
                # Ensure that self.current_price is treated as a scalar
                self.current_price = np.clip(self.current_price.item() + price_change, self.min_price, self.max_price)
            else:
                raise ValueError(f"Invalid action provided for discrete setting: {action}")
        else:
            # If action is continuous but might be an array, take scalar value
            action_value = action if np.isscalar(action) else action.item()
            self.current_price = np.clip(action_value, self.min_price, self.max_price)

        # Update current demand
        self.market_demand = external_market_demand
        self.current_demand = self.demand_calculator.calculate_demand(self.current_price, external_market_demand)

        print(f"Current Demand: {self.current_demand}, Type: {type(self.current_demand)}")
        print(f"Current Price: {self.current_price}, Type: {type(self.current_price)}")

        reward = self.current_demand * self.current_price
        state = np.array([self.current_demand, self.current_price])

        done = False
        revenue = self.current_price * self.current_demand
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
