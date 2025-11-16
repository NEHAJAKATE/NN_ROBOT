import math as m
from sympy import python
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import src
from src.ik_dataset import generate_dataset
from src.nn_ik_model import IKNet


def train():
    X, Y = generate_dataset(3000)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = IKNet()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(15):
        running = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()

        print(f"[TRAIN] Epoch {epoch+1}/15, Loss = {running / len(loader):.6f}")

    torch.save(model.state_dict(), "ik_nn_model.pth")
    print("[TRAIN] Saved model to ik_nn_model.pth")


if __name__ == "__main__":
    train()
