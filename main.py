from data import Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import nn


def get_prepped_data():
    data = pd.read_csv("loan_approval_dataset.csv")

    # prepping data
    data_obj = Data(data)
    data_obj.clean_data()
    data_obj.transform()
    x_train, y_train, x_test, y_test = data_obj.get_data()

    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.from_numpy(y_train.values).float().view(-1, 1)
    x_test = torch.from_numpy(x_test.values).float()
    y_test = torch.from_numpy(y_test.values).float().view(-1, 1)

    return x_train, y_train, x_test, y_test


def plot_loss(losses):
    x_axis_linear = np.arange(len(losses))
    plt.plot(x_axis_linear, losses, label="Linear Regression Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()
    plt.show()


def main():
    # getting datasets
    x_train, y_train, x_test, y_test = get_prepped_data()

    # training the neural network
    model = nn.NeuralNetwork(x_train)

    losses = []
    epochs = 1501
    for epoch in range(epochs):
        # forward pass
        predictions = model.forward(x_train)

        # calculating loss
        loss = nn.bce_loss(predictions, y_train)

        # back propagation
        model.backward(x_train, y_train)
        losses.append(loss.item())

        # loss monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    # evaluating the model
    predictions = model.forward(x_test)
    predictions = torch.where(predictions > 0.5, 1, 0)

    # getting the accuracy of the model
    accuracy = nn.accuracy(predictions, y_test)
    print(f"Accuracy of the Neural Network: {accuracy: .2f}%")

    # plotting losses
    plot_loss(losses)


if __name__ == "__main__":
    main()
