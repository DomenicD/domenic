from typing import Callable, Sequence, Tuple

from python.notebooks.networks import NeuralNetwork


class TrainingPlan:
    def __init__(self,
                 acceptable_error: float = 0,
                 batch_size: int = 100,
                 parameter_recording_interval: int = 10):
        self.acceptable_error = acceptable_error
        self.batch_size = batch_size
        self.parameter_recording_interval = parameter_recording_interval

# TODO: Create training classes that make training, validation, and testing easy. And that make
# automate recording the transformation of network parameters. Keep in mind that eventually you
# will want to make an abstract Trainer with implementations that can be used to train on images,
# time series, and other data types.
class ClosedFormFunctionTrainer:
    def __init__(self, network: NeuralNetwork,
                 function: Callable[[Sequence[float]], Sequence[float]],
                 domain: Tuple[float, float],
                 training_plan: TrainingPlan = TrainingPlan()):
        self.network = network
        self.training_plan = training_plan
        self.function = function
        self.domain = domain

    def single_training_step(self):
        pass

    def batch_train(self):
        pass





