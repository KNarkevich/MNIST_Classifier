from src.cnn.lenet import LeNet

from src.interfaces.digit_classifacation_interface import DigitClassificationInterface


class CNNModel(DigitClassificationInterface):
    def __init__(self, model):
        super(CNNModel, self).__init__(model)
    
    def predict(self, image):

        pred = self.model(image)
        prediction = pred.argmax(axis=1).cpu().numpy()[0]
        return prediction
