from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import torch

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = [data[0].numpy() for data in train_dataset]
    y_train = [data[1] for data in train_dataset]
    X_test = [data[0].numpy() for data in test_dataset]
    y_test = [data[1] for data in test_dataset]

    return X_train, y_train, X_test, y_test

def create_rf_model():
    model = RandomForestClassifier(n_estimators=100)
    return model

def train_rf(model, X_train, y_train):
    model.fit(X_train, y_train)
    torch.save(rf_model, 'models/model_rf.pth')

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    rf_model = create_rf_model()
    train_rf(rf_model, X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
