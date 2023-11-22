import torch
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from digit_classifier import DigitClassifier


def test_for_randome_model():
    
    testData = MNIST(root="data", train=False, download=True,
        transform=ToTensor())

    testDataLoader = DataLoader(testData, batch_size=1)

    preds = []
    labels = []
    randome_model = DigitClassifier('random')
    for (image, label) in testDataLoader:
        pred = randome_model.predict(image)
        preds.append(pred)
        labels.append(label[0])
        
    accuracy = accuracy_score(labels, preds)
    print(f'Accuracy random: {accuracy * 100:.2f}%')


def test_for_random_forest():
    rf_model = DigitClassifier('rf')
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_test = [data[0].numpy() for data in test_dataset]
    y_test = [data[1] for data in test_dataset]

    X_test = np.array(X_test)

    X_test = X_test.reshape(X_test.shape[0], -1)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy RF: {accuracy * 100:.2f}%')
    

def test_cnn():
    cnn_model = DigitClassifier('cnn')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_data = MNIST(root="data", train=False, download=True,
        transform=ToTensor())
    idxs = np.random.choice(range(0, len(test_data)), size=(10,))
    test_data = Subset(test_data, idxs)
    test_loader = DataLoader(test_data, batch_size=1)
    
    preds = []
    labels = []
    for (image, label) in test_loader:
        image = image.to(device)
        pred = cnn_model.predict(image)
        preds.append(pred)
        labels.append(label)

    accuracy = accuracy_score(labels, preds)
    print(f'Accuracy CNN: {accuracy * 100:.2f}%')


if __name__ == "__main__":

    test_cnn()
    test_for_random_forest()
    test_for_randome_model() 
 