import numpy as np

from python.notebooks.utils import tolist


class Parameter:
    def __init__(self, name: str, values: np.ndarray, gradients: np.ndarray):
        self.name = name
        self.values = values
        self.gradients = gradients

    def serialize(self):
        return {
            "name": self.name,
            "values": tolist(self.values),
            "gradients": tolist(self.gradients)
        }
