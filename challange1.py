import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

def f1(x):
    return np.sin(x)

def f2(x):
    return x**2

def f3(x):
    return np.exp(-x**2)

def f4(x):
    return np.sin(3*x) + 0.3*x

functions = {
    "sin(x)": f1,
    "x^2": f2,
    "exp(-x^2)": f3,
    "sin(3x) + 0.3x": f4
}

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

x_train = np.linspace(-3, 3, 200)
x_test = np.linspace(-3, 3, 400)

plt.figure(figsize=(14, 10))

for i, (name, func) in enumerate(functions.items(), 1):
    y_train = func(x_train)
    y_test = func(x_test)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2000):
        optimizer.zero_grad()
        output = model(x_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    y_pred = model(x_test_tensor).detach().numpy().flatten()
    mse = np.mean((y_test - y_pred) ** 2)

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