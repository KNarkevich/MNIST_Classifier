import torch

from src.interfaces.digit_classifacation_interface import DigitClassificationInterface


class RandomModel(DigitClassificationInterface):
    def __init__(self, model):
        super(RandomModel, self).__init__(model)

    def predict(self, image):
        return torch.randint(0, 10, (1,), dtype=torch.long).item()
    