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
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output


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


# TODO: Remove test_y
def data_loader_from_file(train_x, train_y, test_x, test_y):
    train_x = np.loadtxt(train_x, dtype=float)
    train_y = np.loadtxt(train_y, dtype=int)

    train_x_std, train_x_mean = train_x.std(), train_x.mean()
    train_x = (train_x - train_x_mean) / train_x_std

    train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).type(torch.LongTensor))
    train_size = int(0.9 * len(train_x))
    validate_size = len(train_x) - train_size

    test_x = np.loadtxt(test_x, dtype=float)
    test_x = (test_x - train_x_mean) / train_x_std
    test_y = np.loadtxt(test_y, dtype=int)
    test_size = len(test_x)
    test_data = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).type(torch.LongTensor))
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).type(torch.LongTensor))

    train_dataset, validate_dataset = torch.utils.data.random_split(train_data, [train_size, validate_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=False)
    return train_loader, validate_loader, test_loader


def run_model(train_loader, validate_loader, optimizer, model, epochs=10, loss_fn=nn.NLLLoss()):
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
    plt.savefig(f'{plot_title}.png')


def best_model(train_loader, validate_loader, test_loader=None):
    nn_sequential = nn.Sequential(
        nn.Linear(28 * 28, 256),
        nn.Dropout(0.1),
        nn.LeakyReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.Dropout(0.1),
        nn.LeakyReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 10),
        nn.Dropout(0.1),
        nn.LogSoftmax(dim=1)
    )

    model = NeuralNetwork(nn_sequential).to(device)
    gain = nn.init.calculate_gain('leaky_relu')
    # for i, l in enumerate(model.model):
    #     if isinstance(l, nn.Linear):
    #         torch.nn.init.xavier_uniform_(l.weight, gain)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    plot_data = run_model(train_loader, validate_loader, optimizer, model, epochs=10, loss_fn=nn.NLLLoss())
    print(optimizer)

    optimizer.param_groups[0]['lr'] = 5e-4
    plot_data = run_model(train_loader, validate_loader, optimizer, model, epochs=10, loss_fn=nn.NLLLoss())
    print(optimizer)

    optimizer.param_groups[0]['lr'] = 1e-4
    plot_data = run_model(train_loader, validate_loader, optimizer, model, epochs=10, loss_fn=nn.NLLLoss())
    make_plot(plot_data, 'Best Model')
    print(model)
    print(optimizer)


def main():
    train_x, train_y, test_x, test_y = sys.argv[1:]

    train_loader, validate_loader, test_loader = data_loader_from_file(train_x, train_y, test_x, test_y)
    best_model(train_loader, test_loader)
    #
    # # run model A
    # nn_sequential = nn.Sequential(
    #     nn.Linear(28 * 28, 100),
    #     nn.ReLU(),
    #     nn.Linear(100, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, 10),
    #     nn.LogSoftmax(dim=1)
    # )
    # model = NeuralNetwork(nn_sequential).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # plot_data = run_model(train_loader, validate_loader, optimizer, model)
    # make_plot(plot_data, 'Model A')
    #
    # # run model B
    # nn_sequential = nn.Sequential(
    #     nn.Linear(28 * 28, 100),
    #     nn.ReLU(),
    #     nn.Linear(100, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, 10),
    #     nn.LogSoftmax(dim=1)
    # )
    # model = NeuralNetwork(nn_sequential).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    # plot_data = run_model(train_loader, validate_loader, optimizer, model)
    # make_plot(plot_data, 'Model B')
    #
    # nn_sequential = nn.Sequential(
    #     nn.Linear(28 * 28, 100),
    #     nn.Dropout(0.25),
    #     nn.ReLU(),
    #     nn.Linear(100, 50),
    #     nn.Dropout(0.25),
    #     nn.ReLU(),
    #     nn.Linear(50, 10),
    #     nn.Dropout(0.25),
    #     nn.LogSoftmax(dim=1)
    # )
    # # run model C
    # model = NeuralNetwork(nn_sequential).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    # plot_data = run_model(train_loader, validate_loader, optimizer, model)
    # make_plot(plot_data, 'Model C')
    #
    # # run model D (Batch Norm before activation)
    # nn_sequential = nn.Sequential(
    #     nn.Linear(28 * 28, 100),
    #     nn.BatchNorm1d(100),
    #     nn.ReLU(),
    #     nn.Linear(100, 50),
    #     nn.BatchNorm1d(50),
    #     nn.ReLU(),
    #     nn.Linear(50, 10),
    #     nn.LogSoftmax(dim=1)
    # )
    # model = NeuralNetwork(nn_sequential).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    # plot_data = run_model(train_loader, validate_loader, optimizer, model)
    # make_plot(plot_data, 'Model D - Batch Norm before activation')
    #
    # # run model D (Batch Norm after activation)
    # nn_sequential = nn.Sequential(
    #     nn.Linear(28 * 28, 100),
    #     nn.ReLU(),
    #     nn.BatchNorm1d(100),
    #     nn.Linear(100, 50),
    #     nn.ReLU(),
    #     nn.BatchNorm1d(50),
    #     nn.Linear(50, 10),
    #     nn.LogSoftmax(dim=1)
    # )
    # model = NeuralNetwork(nn_sequential).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    # plot_data = run_model(train_loader, validate_loader, optimizer, model)
    # make_plot(plot_data, 'Model D - Batch Norm after activation')
    #
    # # run model E
    # nn_sequential = nn.Sequential(
    #     nn.Linear(28 * 28, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 10),
    #     nn.ReLU(),
    #     nn.Linear(10, 10),
    #     nn.ReLU(),
    #     nn.Linear(10, 10),
    #     nn.ReLU(),
    #     nn.Linear(10, 10),
    #     nn.LogSoftmax(dim=1)
    # )
    # model = NeuralNetwork(nn_sequential).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    # plot_data = run_model(train_loader, validate_loader, optimizer, model)
    # make_plot(plot_data, 'Model E')
    #
    # # run model F
    # nn_sequential = nn.Sequential(
    #     nn.Linear(28 * 28, 128),
    #     nn.Sigmoid(),
    #     nn.Linear(128, 64),
    #     nn.Sigmoid(),
    #     nn.Linear(64, 10),
    #     nn.Sigmoid(),
    #     nn.Linear(10, 10),
    #     nn.Sigmoid(),
    #     nn.Linear(10, 10),
    #     nn.Sigmoid(),
    #     nn.Linear(10, 10),
    #     nn.LogSoftmax(dim=1)
    # )
    # model = NeuralNetwork(nn_sequential).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    # plot_data = run_model(train_loader, validate_loader, optimizer, model)
    # make_plot(plot_data, 'Model F')
    #


if __name__ == '__main__':
    main()
