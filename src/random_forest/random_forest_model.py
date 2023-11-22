import numpy as np
from src.interfaces.digit_classifacation_interface import DigitClassificationInterface


class RFModel(DigitClassificationInterface):
    def __init__(self, model):
        super(RFModel, self).__init__(model)
        
    def predict(self, image):
        pred = self.model.predict(image)
        
        return pred

