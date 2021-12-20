import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = model

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)

        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= len(dataloader)
    correct *= (100 / size)
    print(f"Train Error: \n Accuracy: {(correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return train_loss, correct


def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    validate_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validate_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    validate_loss /= num_batches
    correct *= (100 / size)
    print(f"Validate Error: \n Accuracy: {(correct):>0.1f}%, Avg loss: {validate_loss:>8f} \n")
    return validate_loss, correct


def data_loader_from_file(train_x,train_y):
    train_x = np.loadtxt(train_x, dtype=float)
    train_y = np.loadtxt(train_y, dtype=int)
    train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).type(torch.LongTensor))
    train_size = int(0.8 * len(train_x))
    test_size = len(train_x) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=False)
    return train_loader, validate_loader


def run_model(train_loader, validate_loader, optimizer, model):
    loss_fn = nn.NLLLoss()
    epochs = 10
    train_losses, train_corrects = [], []
    validate_losses, validate_corrects = [], []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss, train_correct = train(train_loader, model, loss_fn, optimizer)
        train_losses.append(train_loss)
        train_corrects.append(train_correct)
        validate_loss, validate_correct = validate(validate_loader, model, loss_fn)
        validate_losses.append(validate_loss)
        validate_corrects.append(validate_correct)
    print("Done!")
    return {"train_loss": train_losses, "train_acc": train_corrects, "validate_loss": validate_losses,
            "validate_acc": validate_corrects}


# def run_model_a(train_loader, validate_loader):
#     model = NeuralNetworkModel().to(device)
#     loss_fn = nn.NLLLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     epochs = 10
#     train_losses, train_corrects = [], []
#     validate_losses, validate_corrects = [], []
#     for t in range(epochs):
#         print(f"Epoch {t + 1}\n-------------------------------")
#         train_loss, train_correct = train(train_loader, model, loss_fn, optimizer)
#         train_losses.append(train_loss)
#         train_corrects.append(train_correct)
#         validate_loss, validate_correct = validate(validate_loader, model, loss_fn)
#         validate_losses.append(validate_loss)
#         validate_corrects.append(validate_correct)
#     print("Done!")
#     return {"train_loss": train_losses, "train_acc": train_corrects, "validate_loss": validate_losses,
#             "validate_acc": validate_corrects}
#
#
# def run_model_b(train_loader, validate_loader):
#     model = NeuralNetworkModel().to(device)
#     loss_fn = nn.NLLLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,betas=(0.9,0.999),eps=1e-8)
#     epochs = 10
#     train_losses, train_corrects = [], []
#     validate_losses, validate_corrects = [], []
#     for t in range(epochs):
#         print(f"Epoch {t + 1}\n-------------------------------")
#         train_loss, train_correct = train(train_loader, model, loss_fn, optimizer)
#         train_losses.append(train_loss)
#         train_corrects.append(train_correct)
#         validate_loss, validate_correct = validate(validate_loader, model, loss_fn)
#         validate_losses.append(validate_loss)
#         validate_corrects.append(validate_correct)
#     print("Done!")
#     return {"train_loss": train_losses, "train_acc": train_corrects, "validate_loss": validate_losses,
#             "validate_acc": validate_corrects}
#
# def run_model_c(train_loader, validate_loader):
#     model = NeuralNetworkModel(dropout=True).to(device)
#     loss_fn = nn.NLLLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,betas=(0.9,0.999),eps=1e-8)
#     epochs = 10
#     train_losses, train_corrects = [], []
#     validate_losses, validate_corrects = [], []
#     for t in range(epochs):
#         print(f"Epoch {t + 1}\n-------------------------------")
#         train_loss, train_correct = train(train_loader, model, loss_fn, optimizer)
#         train_losses.append(train_loss)
#         train_corrects.append(train_correct)
#         validate_loss, validate_correct = validate(validate_loader, model, loss_fn)
#         validate_losses.append(validate_loss)
#         validate_corrects.append(validate_correct)
#     print("Done!")
#     return {"train_loss": train_losses, "train_acc": train_corrects, "validate_loss": validate_losses,
#             "validate_acc": validate_corrects}

def make_plot(plots_data, plot_title):
    train_loss = plots_data["train_loss"]
    train_acc = plots_data["train_acc"]
    validate_loss = plots_data["validate_loss"]
    validate_acc = plots_data["validate_acc"]

    fix, axes = plt.subplots(1, 2, figsize=(15, 5))

    plt.suptitle(plot_title)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch #")
    axes[0].set_ylabel("Loss")
    axes[0].plot(train_loss, label="Train")
    axes[0].plot(validate_loss, label="Validate")
    axes[0].legend()

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch #")
    axes[1].set_ylabel("Accuracy %")
    axes[1].plot(train_acc, label="Train")
    axes[1].plot(validate_acc, label="Validate")
    axes[1].legend()

    plt.show()


def main():
    train_x, train_y, test_x, test_y = sys.argv[1:]
    train_loader, validate_loader = data_loader_from_file(train_x,train_y)

    # run model A
    nn_sequential = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.LogSoftmax(dim=1)
    )
    model = NeuralNetwork(nn_sequential).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    plot_data = run_model(train_loader, validate_loader, optimizer, model)
    make_plot(plot_data, 'Model A')

    # run model B
    model = NeuralNetwork(nn_sequential).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    plot_data = run_model(train_loader, validate_loader, optimizer, model)
    make_plot(plot_data, 'Model B')

    nn_sequential = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.Dropout(0.25),
        nn.LogSoftmax(dim=1)
    )
    # run model C
    model = NeuralNetwork(nn_sequential).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    plot_data = run_model(train_loader, validate_loader, optimizer, model)
    make_plot(plot_data, 'Model C')

    # run model D (Batch Norm before activation)
    nn_sequential = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.BatchNorm1d(50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.LogSoftmax(dim=1)
    )
    model = NeuralNetwork(nn_sequential).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    plot_data = run_model(train_loader, validate_loader, optimizer, model)
    make_plot(plot_data, 'Model D - Batch Norm before activation')

    # run model D (Batch Norm after activation)
    nn_sequential = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.ReLU(),
        nn.BatchNorm1d(100),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.BatchNorm1d(50),
        nn.Linear(50, 10),
        nn.LogSoftmax(dim=1)
    )
    model = NeuralNetwork(nn_sequential).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    plot_data = run_model(train_loader, validate_loader, optimizer, model)
    make_plot(plot_data, 'Model D - Batch Norm after activation')

    # run model E
    nn_sequential = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.LogSoftmax(dim=1)
    )
    model = NeuralNetwork(nn_sequential).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    plot_data = run_model(train_loader, validate_loader, optimizer, model)
    make_plot(plot_data, 'Model E')

    # run model F
    nn_sequential = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.LogSoftmax(dim=1)
    )
    model = NeuralNetwork(nn_sequential).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    plot_data = run_model(train_loader, validate_loader, optimizer, model)
    make_plot(plot_data, 'Model F')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
