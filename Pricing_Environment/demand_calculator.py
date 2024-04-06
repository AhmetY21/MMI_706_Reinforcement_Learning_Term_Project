class DemandCalculator:
    def __init__(self, price_probability_ranges):
        """
        Initialize the DemandCalculator with price ranges and associated probabilities.
        :param price_probability_ranges: A dictionary where keys are tuples representing price ranges (min_price, max_price),
                                         and values are probabilities associated with those ranges.
        """
        self.price_probability_ranges = price_probability_ranges
        
    def calculate_demand(self, price, base_demand):
        """
        Calculate demand based on the current price and its corresponding probability.
        :param price: The current price of the product.
        :param base_demand: The base demand level before adjustment.
        :return: The adjusted demand based on the price segment's probability.
        """
        for price_range, probability in self.price_probability_ranges.items():
            if price_range[0] <= price <= price_range[1]:
                return base_demand * probability
        # Default to base demand if no price range matches
        return base_demand