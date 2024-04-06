import numpy as np

class ActionStrategy:
    def __init__(self, actions):
        """
        Initialize the ActionStrategy with a dictionary of actions and probabilities.
        :param actions: A dictionary where keys are action descriptions and values are the probabilities of those actions.
        """
        self.actions = actions
    
    def choose_action(self):
        """
        Chooses an action based on the defined probabilities.
        :return: An action key as defined in the actions dictionary.
        """
        actions, probabilities = zip(*self.actions.items())
        return np.random.choice(actions, p=probabilities)