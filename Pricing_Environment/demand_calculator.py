import numpy as np

class DemandDataGenerator:
    def __init__(self, low, high, steps, seasonality=False):
        """
        Initialize the demand data generator for a single set of steps.

        Args:
        low (int): The minimum demand value.
        high (int): The maximum demand value.
        steps (int): Number of time steps for which to generate data.
        seasonality (bool): Whether to include seasonality in the demand data.
        """
        self.low = low
        self.high = high
        self.steps = steps
        self.seasonality = seasonality

    def generate(self):
        """
        Generates demand data based on initialized settings.

        Returns:
        np.array: Demand data for the defined number of steps.
        """
        if self.seasonality:
            # Generate seasonal data using a sinusoidal pattern
            t = np.linspace(0, 2 * np.pi, self.steps)
            seasonal_effect = 10 * np.sin(t)
            data = np.random.randint(self.low, self.high, self.steps) + seasonal_effect
        else:
            data = np.random.randint(self.low, self.high, self.steps)
        return data


class DemandCalculator:
    def __init__(self, price_probability_ranges, base_price=10, elasticity=-1.5):
        """
        Initialize the DemandCalculator with discrete price ranges, associated probabilities,
        and parameters for continuous demand calculation using elasticity.

        :param price_probability_ranges: A dictionary where keys are tuples representing discrete price ranges (min_price, max_price),
                                         and values are probabilities associated with those ranges.
        :param base_price: The reference price for elasticity calculations.
        :param elasticity: The price elasticity of demand.
        """
        self.price_probability_ranges = price_probability_ranges
        self.base_price = base_price
        self.elasticity = elasticity

    def _calculate_demand_continuous(self, price, market_demand):
        """
        Calculate demand continuously based on price elasticity.

        :param price: The current price of the product.
        :param market_demand: The current market demand.
        :return: The adjusted demand based on elasticity.
        """
        # Adjust the demand based on elasticity formula and market demand
        return market_demand * ((price / self.base_price) ** self.elasticity)

    def _calculate_demand_discrete(self, price, market_demand):
        """
        Calculate demand discretely based on predefined price ranges and probabilities.

        :param price: The current price of the product.
        :param market_demand: The current market demand.
        :return: The adjusted demand based on the price segment's probability.
        """
        for price_range, probability in self.price_probability_ranges.items():
            if price_range[0] <= price < price_range[1]:
                return market_demand * probability
        return market_demand

    def calculate_demand(self, price, market_demand, method="discrete"):
        """
        Calculate demand based on the method specified ('continuous' or 'discrete') and current market demand.

        :param price: The current price of the product.
        :param market_demand: The current market demand.
        :param method: The calculation method to use ('continuous' or 'discrete').
        :return: The adjusted demand based on the selected method and market conditions.
        """
        if method == "continuous":
            return self._calculate_demand_continuous(price, market_demand)
        elif method == "discrete":
            return self._calculate_demand_discrete(price, market_demand)
        else:
            raise ValueError("Invalid method specified. Use 'continuous' or 'discrete'.")
