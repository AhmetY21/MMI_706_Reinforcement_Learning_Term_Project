import numpy as np
import random

random.seed(42)
np.random.seed(42)

class DemandDataGenerator:
    def __init__(self, low, high, steps, seasonality=False, seasonal_amplitude=10, seasonal_frequency=1, phase_shift=0):
        self.low = low
        self.high = high
        self.steps = steps
        self.seasonality = seasonality
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_frequency = seasonal_frequency
        self.phase_shift = phase_shift

    def generate(self):
        if self.seasonality:
            t = np.linspace(0, 2 * np.pi, self.steps)
            seasonal_effect = self.seasonal_amplitude * np.sin(self.seasonal_frequency * t + self.phase_shift)
            data = np.random.randint(self.low, self.high, self.steps) + seasonal_effect
        else:
            data = np.random.randint(self.low, self.high, self.steps)
        return data

class DemandCalculator:
    def __init__(self, price_probability_ranges, base_price=10, elasticity=-1.5, price_caps=None, cap_start_step=0):
        self.price_probability_ranges = price_probability_ranges
        self.base_price = base_price
        self.elasticity = elasticity
        self.price_caps = price_caps
        self.cap_start_step = cap_start_step

    def _apply_price_caps(self, price, current_step):

        if self.price_caps and current_step >= self.cap_start_step:
            min_cap, max_cap = self.price_caps
            return max(min(price, max_cap), min_cap)
        return price

    def _calculate_demand_continuous(self, price, market_demand):
        return market_demand * ((price / self.base_price) ** self.elasticity)

    def _calculate_demand_discrete(self, price, market_demand):
        for price_range, probability in self.price_probability_ranges.items():
            if price_range[0] <= price < price_range[1]:
                return market_demand * probability
        return market_demand
    def effective_price(self,price,current_step):
        return self._apply_price_caps(price, current_step)
        
    def calculate_demand(self, price, market_demand, competitor_price=None, method="discrete", is_monopoly=False, current_step=0):
        """
        Calculate demand considering competitor's pricing unless operating as a monopoly. 
        Apply price caps if they are set and the current step is beyond the cap start step.

        :param price: The price set by our product.
        :param market_demand: The total market demand.
        :param competitor_price: The price set by a competitor.
        :param method: The calculation method ('continuous' or 'discrete').
        :param is_monopoly: Boolean indicating whether the market is a monopoly.
        :param current_step: The current time step in the simulation.
        :return: The adjusted demand considering competitor influence and price caps.
        """
        effective_price = self._apply_price_caps(price, current_step)
        effective_competitor_price = self._apply_price_caps(competitor_price, current_step)

        if is_monopoly:
            if method == "continuous":
                return self._calculate_demand_continuous(effective_price, market_demand)
            elif method == "discrete":
                return self._calculate_demand_discrete(effective_price, market_demand)
            else:
                raise ValueError("Invalid method specified. Use 'continuous' or 'discrete'.")

        if method == "continuous":
            demand = self._calculate_demand_continuous(effective_price, market_demand)
        elif method == "discrete":
            demand = self._calculate_demand_discrete(effective_price, market_demand)
        else:
            raise ValueError("Invalid method specified. Use 'continuous' or 'discrete'.")

        if effective_competitor_price < effective_price:
            competitor_demand = self._calculate_demand_continuous(effective_competitor_price, market_demand) if method == "continuous" \
                else self._calculate_demand_discrete(effective_competitor_price, market_demand)
            demand -= competitor_demand
        elif effective_competitor_price == effective_price:
            demand *= 0.5
        elif effective_competitor_price > effective_price:
            ratio = effective_price / effective_competitor_price
            demand *= (1 - ratio)

        return max(demand, 0)  # Ensure demand does not go negative
