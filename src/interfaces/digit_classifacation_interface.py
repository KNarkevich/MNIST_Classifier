
class DigitClassificationInterface:
    def __init__(self, model):
        self.model = model

    def predict(self, input_features):

        raise NotImplementedError

    def predict_batch(self, batch_input_features):
        
        raise NotImplementedError
    
    def train(self, dataset):

        raise NotImplementedError
    