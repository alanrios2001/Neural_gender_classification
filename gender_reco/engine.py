import torch
from tqdm.auto import tqdm
from helpers import acc_fn


def train_step(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: torch.device,
               loaded_train: torch.utils.data.DataLoader):

    train_loss = 0
    train_acc = 0

    model.train()
    for X, y in tqdm(loaded_train, ascii="123456789$"):
        y = y.to(device)

        y_preds = model(X).squeeze(1)

        loss = loss_fn(y_preds, y.float())
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = acc_fn(y_true=y, y_pred=torch.round(torch.sigmoid(y_preds)))
        train_acc += acc
    train_loss /= len(loaded_train)
    train_acc /= len(loaded_train)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              device: torch.device,
              loaded_test: torch.utils.data.DataLoader):
    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(loaded_test, ascii="123456789$"):
            y = y.to(device)

            y_preds = model(X).squeeze(1)
            loss = loss_fn(y_preds, y.float())
            test_loss += loss.item()
            acc = acc_fn(y_true=y, y_pred=torch.round(torch.sigmoid(y_preds)))
            test_acc += acc
        test_loss /= len(loaded_test)
        test_acc /= len(loaded_test)

    return test_loss, test_acc


def train(epochs: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          device: torch.device,
          loaded_train: torch.utils.data.DataLoader,
          loaded_test: torch.utils.data.DataLoader,
          classes: list):

    results = {"epoch": [],
               "train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               "classes": classes,
               }

    for epoch in range(epochs):

        train_loss, train_acc = train_step(model, optimizer, loss_fn, device, loaded_train)

        test_loss, test_acc = test_step(model, loss_fn, device, loaded_test)

        results["epoch"].append(epoch)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(f"train loss: {train_loss:.4f} || test loss: {test_loss:.4f} || train acc: {train_acc:.2f}% || test acc: {test_acc:.2f}%")

    return results
