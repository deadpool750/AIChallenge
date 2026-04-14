import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

#setting a seed so that random numbers are repeatable
torch.manual_seed(42)
np.random.seed(42)


#functions for the neural network to learn
def f1(x):
    return np.sin(x)

def f2(x):
    return x**2

def f3(x):
    return np.exp(-x**2)

def f4(x):
    return np.sin(3*x) + 0.3*x


#dictionary storing function names and their actual formulas
functions = {
    "sin(x)": f1,
    "x^2": f2,
    "exp(-x^2)": f3,
    "sin(3x) + 0.3x": f4
}


#neural network model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Structure:
        # 1 input value x
        #   first hidden layer with 32 neurons
        #   Tanh activation
        #   second hidden layer with 32 neurons
        #   Tanh activation
        #   final output layer with 1 value y

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


#training x-values 200 spaced points from -3 to 3
x_train = np.linspace(-3, 3, 200)

#testing x-values 400 points for a smoother final graph
x_test = np.linspace(-3, 3, 400)

#one figure to for the 4 plots
plt.figure(figsize=(14, 10))


#loop through each function
for i, (name, func) in enumerate(functions.items(), 1):

    #computing the true y-values for training and testing
    y_train = func(x_train)
    y_test = func(x_test)

    #converting NumPy arrays into PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)

    #fresh neural network for this function
    model = MLP()

    #loss function:
    #measures how far predictions are from the correct answers
    criterion = nn.MSELoss()

    #optimizer
    #lr = learning rate = how big each update step should be
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #training for 2000 epochs
    for epoch in range(2000):

        #remove old stored gradients
        #gradients tell the model how to change its weights
        optimizer.zero_grad()

        #forward pass:
        #feed training x values into the model
        #model outputs its current guesses for y
        output = model(x_train_tensor)

        #compare models guesses to the real values
        loss = criterion(output, y_train_tensor)

        #backward pass:
        #PyTorch calculates gradients automatically here.
        #it figures out how each weight in the network contributed to the final error.
        loss.backward()

        #update step:
        #adam uses the gradients to slightly change the weights
        optimizer.step()

    #after training, use the model on test points
    y_pred = model(x_test_tensor).detach().numpy().flatten()

    #computing final mean squared error on test data
    mse = np.mean((y_test - y_pred) ** 2)

    #plots
    plt.subplot(2, 2, i)
    plt.plot(x_test, y_test, label="True function")
    plt.plot(x_test, y_pred, "--", label="MLP approximation")
    plt.scatter(x_train, y_train, s=10, alpha=0.4, label="Training points")
    plt.title(f"{name}\nMSE = {mse:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()