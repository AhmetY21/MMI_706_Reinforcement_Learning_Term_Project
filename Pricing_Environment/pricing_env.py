import gymnasium
from gymnasium import spaces
import numpy as np
from Pricing_Environment import demand_calculator
from Pricing_Environment.demand_calculator import DemandDataGenerator, DemandCalculator  # Assume correct imports based on your file structure

class PricingEnvironment(gymnasium.Env):
    metadata = {'render_modes': ['text', 'rgb_array'], 'render_fps': 4}

    def __init__(self, product_config, demand_calculator_config, action_strategy_config, is_continuous=False, render_mode=None):
        super().__init__()
        self.min_price = product_config["min_price"]
        self.max_price = product_config["max_price"]
        self.current_demand = product_config["initial_demand"]
        self.current_price = np.mean([self.min_price, self.max_price])
        self.step_count = 0  # Keep track of simulation steps
        self.action_strategy_config = action_strategy_config
        
        # Initialize the demand generator
        self.demand_generator = DemandDataGenerator(
            low=demand_calculator_config['low'],
            high=demand_calculator_config['high'],
            steps=product_config['steps'],
            seasonality=demand_calculator_config['seasonality'],
            seasonal_amplitude=demand_calculator_config.get('seasonal_amplitude', 10),
            seasonal_frequency=demand_calculator_config.get('seasonal_frequency', 1),
            phase_shift=demand_calculator_config.get('phase_shift', 0)
        )

        # Initialize the demand calculator
        self.demand_calculator = DemandCalculator(
            price_probability_ranges=demand_calculator_config["price_probability_ranges"],
            base_price=demand_calculator_config.get('base_price', 10),
            elasticity=demand_calculator_config.get('elasticity', -1.5),
            price_caps=demand_calculator_config.get('price_caps', None),
            cap_start_step=demand_calculator_config.get('cap_start_step', 0)
        )

        # Action space definition
        self.is_continuous = is_continuous
        if is_continuous:
            self.action_space = spaces.Box(low=np.array([self.min_price]), high=np.array([self.max_price]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(action_strategy_config["price_change_map"]))

        self.observation_space = spaces.Box(low=np.array([0, self.min_price]), high=np.array([np.inf, self.max_price]), dtype=np.float32)
        
        self.render_mode = render_mode
        self.history = []

    def step(self, action):
        # Increment step count for price cap checks
        self.step_count += 1

        # Handle price change based on action input
        if not self.is_continuous:
            price_change = self.action_strategy_config["price_change_map"].get(action)
            if price_change is None:
                raise ValueError(f"Invalid action provided for discrete setting: {action}")
            self.current_price = np.clip(self.current_price + price_change, self.min_price, self.max_price)
            self.current_price = self.demand_calculator.effective_price(self.current_price,self.step_count)
        else:
            action_value = action if np.isscalar(action) else action.item()
            self.current_price = np.clip(action_value, self.min_price, self.max_price)


        self.current_price = self.demand_calculator.effective_price(self.current_price, self.step_count)

        # Generate external market demand
        external_market_demand = self.demand_generator.generate()[self.step_count % self.demand_generator.steps]  # Circular indexing if steps > total steps in simulation
        competitor_price = np.random.uniform(self.min_price, self.max_price)  # Simulate a dynamic competitor price

        # Update current demand
        self.current_demand = self.demand_calculator.calculate_demand(
            self.current_price,
            external_market_demand,
            competitor_price,
            'discrete',
            False,
            self.step_count
        )

        reward = self.current_demand * self.current_price
        state = np.array([self.current_demand, self.current_price])

        done = False  # Update if there are specific termination conditions
        self.history.append((self.current_price, external_market_demand, self.current_demand, self.current_demand * self.current_price))
        
        return state, reward, done, {}

    def reset(self):
        self.current_price = np.mean([self.min_price, self.max_price])
        self.market_demand = self.demand_generator.generate()[0]  # Start with the first demand in the cycle
        self.history = []
        self.step_count = 0
        return np.array([self.market_demand, self.current_price])

    def render(self, mode='text'):
        if mode == 'text':
            latest_record = self.history[-1] if self.history else (self.current_price, self.market_demand, self.current_demand)
            print(f"Price: {latest_record[0]}, Market Demand: {latest_record[1]}, Current Demand: {latest_record[2]}")

    def close(self):
        pass

# Example configuration dictionaries might be defined here or loaded from a config file.
