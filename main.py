import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 100
K = 3
X = torch.zeros((N*K, 2))
Y = torch.zeros(N*K, dtype=torch.long)

for j in range(K):
    i = range(j * N, N * (j + 1))
    r = torch.linspace(0, 1, N)
    t = torch.linspace((j * (K + 1)), (j + 1) * (K + 1), N) + torch.randn(N) * 0.2
    X[i] = torch.cat((r.unsqueeze(1) * torch.cos(t).unsqueeze(1), r.unsqueeze(1) * torch.sin(t).unsqueeze(1)), dim=1)
    Y[i] = j

class SpiralClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 128)
        self.layer_2 = nn.Linear(128, 3)

    def forward(self, x):
        out = torch.relu(self.layer_1(x))
        out = self.layer_2(out)
        return out

model = SpiralClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epochs = 600

#Provided by an AI system for visualisation of the task
def plot_decision_boundary(model, x, y, epoch):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    x_axis = np.arange(x_min, x_max, 0.1)
    y_axis = np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x_axis, y_axis)

    grid_points_np = np.c_[xx.ravel(), yy.ravel()]
    grid_points_torch = torch.from_numpy(grid_points_np).float()

    with torch.no_grad():
        predictions = model(grid_points_torch)

    # Find the index of the highest score for each point
    final_predictions = torch.max(predictions, dim=1)[1]

    predictions_grid = final_predictions.view(xx.shape)
    plt.contourf(xx, yy, predictions_grid.detach().numpy(), cmap='coolwarm', alpha=0.5)
    plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), c=y.detach().numpy(), cmap='coolwarm', edgecolor='black')
    plt.savefig(f"images/epoch-{epoch}.png")

for epoch in range(epochs):
    y_pred = model(X)

    loss = criterion(y_pred, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % (epochs /100) == 0:
        plot_decision_boundary(model, X, Y, epoch)
        print(f"Completed epoch {epoch}/{epochs} ; Loss {loss.item()}")


