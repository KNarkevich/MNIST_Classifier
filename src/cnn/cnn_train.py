import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import classification_report
from lenet import LeNet

import time


INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(train_data, test_data):
    
    numTrainSamples = int(len(train_data) * TRAIN_SPLIT)
    numValSamples = int(len(train_data) * VAL_SPLIT)
    (train_data, val_data) = random_split(train_data,
        [numTrainSamples, numValSamples],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, shuffle=True,
        batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    return train_loader, test_loader, val_loader, train_data, val_data

def create_model(train_data):
    model = LeNet(
	    numChannels=1,
	    classes=len(train_data.dataset.classes)).to(device)
    return model

def train(model, train_loader, val_loader, epochs=EPOCHS, lr=INIT_LR):
    
    trainSteps = len(train_loader.dataset) // BATCH_SIZE
    valSteps = len(val_loader.dataset) // BATCH_SIZE
    opt = Adam(model.parameters(), lr)
    lossFn = nn.NLLLoss()

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    print("Training the network...")
    startTime = time.time()

    for e in range(0, epochs):
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0
        for (x, y) in train_loader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = lossFn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        with torch.no_grad():
            model.eval()
            for (x, y) in val_loader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
    
        trainCorrect = trainCorrect / len(train_loader.dataset)
        valCorrect = valCorrect / len(val_loader.dataset)

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
    
        print("EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.5f}, Train accuracy: {:.5f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.5f}, Val accuracy: {:.5f}\n".format(
            avgValLoss, valCorrect))
    
    
    endTime = time.time()
    print("[Total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    torch.save(model, 'models/cnn_model.pth')
    torch.save(model.state_dict(), 'models/cnn_model2.pth')
 
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')
    
    return model
    
def evol(model, test_loader, test_data):
    with torch.no_grad():
        model.eval()
        preds = []
        for (x, y) in test_loader:

            x = x.to(device)
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    print(classification_report(test_data.targets.cpu().numpy(),
        np.array(preds), target_names=test_data.classes))

    
if __name__ == "__main__":
    train_data = MNIST(root="data", train=True, download=True,
	    transform=ToTensor())
    test_data = MNIST(root="data", train=False, download=True,
        transform=ToTensor())
    train_loader, test_loader, val_loader, train_data, val_data = load_data(train_data, test_data)
    model = create_model(train_data)
    trained_model = train(model, train_loader, val_loader)
    evol(trained_model, test_loader, test_data)
    
    