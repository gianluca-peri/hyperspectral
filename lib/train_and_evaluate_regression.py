import time
import torch
import numpy as np


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0

    start_time = time.time()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # Loss is averaged over the batch.
        batch_mae = torch.mean(torch.abs(outputs - targets)).item()
        running_mae += batch_mae

    end_time = time.time()
    epoch_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(epoch_time))

    avg_loss = running_loss / len(train_loader)
    avg_mae = running_mae / len(train_loader)

    return avg_loss, avg_mae, formatted_time


def evaluate(model, evaluation_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_mae = 0.0

    with torch.no_grad():
        for inputs, targets in evaluation_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()  # Loss is averaged over the batch.
            val_mae += torch.mean(torch.abs(outputs - targets)).item()

    avg_val_loss = val_loss / len(evaluation_loader)
    avg_val_mae = val_mae / len(evaluation_loader)

    return avg_val_loss, avg_val_mae


def train_one_epoch_and_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs,
    verbose=False,
):
    train_loss, train_mae, epoch_time = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_mae = evaluate(model, val_loader, criterion, device)

    if verbose:
        print(
            f"Epoch {epoch + 1}/{total_epochs}: "
            f"Train Loss: {train_loss:.2f}, Train MAE: {train_mae:.2f}, "
            f"Val Loss: {val_loss:.2f}, Val MAE: {val_mae:.2f}, Time: {epoch_time}"
        )

    return train_loss, train_mae, val_loss, val_mae


def train_and_evaluate_regression(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    epochs,
    device,
    scheduler_warmup=None,
    scheduler_plateau=None,
    verbose=False,
):
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "learning_rate": [],
    }
    final_test_evaluation = {
        "test_loss": 0.0,
        "test_mae": 0.0,
    }

    if verbose:
        print(f"Starting the training of {model.__class__.__name__}...")

    for epoch in range(epochs):
        train_loss, train_mae, val_loss, val_mae = train_one_epoch_and_evaluate(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epoch,
            epochs,
            verbose,
        )

        # Step schedulers: warmup for the first warmup_epochs, plateau afterwards.
        if scheduler_warmup and epoch < scheduler_warmup.total_iters:
            scheduler_warmup.step()
        elif scheduler_plateau:
            scheduler_plateau.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["learning_rate"].append(current_lr)

    # Final evaluation on the test set.
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    final_test_evaluation["test_loss"] = test_loss
    final_test_evaluation["test_mae"] = test_mae

    return history, final_test_evaluation, model

def reconstruction(test_loader, model, device):
    model.eval()
    reconstructions = []
    true_profile = []

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            reconstructions.append(outputs.cpu().numpy())
            true_profile.append(target.cpu().numpy())
        
    reconstructions = np.concatenate(reconstructions, axis=0)
    true_profile = np.concatenate(true_profile, axis=0)
    inputs = np.concatenate([inputs.cpu().numpy() for inputs, _ in test_loader], axis=0)

    return inputs, true_profile, reconstructions