import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_model(x_train, y_train, hidden_size=32, epochs=2000, lr=0.01):
    model = MLP(hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x_tensor)
        loss = criterion(output, y_tensor)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses


def evaluate_model(model, func, x_test):
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1).to(device)
    with torch.no_grad():
        y_pred = model(x_tensor).cpu().numpy().flatten()

    y_true = func(x_test)
    mse = np.mean((y_true - y_pred)**2)
    return y_true, y_pred, mse


x_train = np.linspace(-3, 3, 200)
x_test = np.linspace(-3, 3, 400)

results = {}

plt.figure(figsize=(14, 10))

for i, (name, func) in enumerate(functions.items(), 1):
    y_train = func(x_train)

    model, losses = train_model(x_train, y_train, hidden_size=32, epochs=2000, lr=0.01)
    y_true, y_pred, mse = evaluate_model(model, func, x_test)

    results[name] = mse

    plt.subplot(2, 2, i)
    plt.plot(x_test, y_true, label="True function")
    plt.plot(x_test, y_pred, "--", label="MLP approximation")
    plt.scatter(x_train, y_train, s=10, alpha=0.4, label="Training points")
    plt.title(f"{name}\nMSE = {mse:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()

print("Final MSE results:")
for name, mse in results.items():
    print(f"{name}: {mse:.6f}")