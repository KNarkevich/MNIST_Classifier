from src.cnn.cnn_model import CNNModel
from src.random_forest.random_forest_model import RFModel
from src.random_model.random_model import RandomModel
from src.cnn.lenet import LeNet

import torch


class DigitClassifier:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.model = self._get_model()

    def _get_model(self):
        if self.algorithm == 'cnn':

            loaded_state_dict = torch.load('models/cnn_model.pth')
            model = LeNet(numChannels=1, classes=10)
            model.load_state_dict(loaded_state_dict)

            cnn_model = CNNModel(model)
            return cnn_model
        
        elif self.algorithm == 'rf':
            
            model = torch.load('models/model_rf.pth')
            rf_model = RFModel(model)
            return rf_model
        
        elif self.algorithm == 'random':
            model = None
            rand_model = RandomModel(model)
            return rand_model
        else:
            raise ValueError("Invalid algorithm. Supported values are 'cnn', 'rf', 'random'.")

    def predict(self, image):
        return self.model.predict(image)
    

