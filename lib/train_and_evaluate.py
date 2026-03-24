import torch
import time

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() # The loss is averaged over the batch
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    end_time = time.time()
    epoch_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(epoch_time))

    avg_loss = running_loss / len(train_loader) # Average loss per batch
    accuracy = 100 * correct / total

    return avg_loss, accuracy, formatted_time

def evaluate(model, evaluation_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    total_confidence = 0.0
    
    with torch.no_grad():
        for inputs, labels in evaluation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() # The loss is averaged over the batch
            
            # Calculate probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            # Get the highest probability (confidence) and the predicted class
            confidence, predicted = torch.max(probs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_confidence += confidence.sum().item()

    avg_val_loss = val_loss / len(evaluation_loader)
    val_acc = 100 * correct / total
    avg_val_confidence = total_confidence / total

    return avg_val_loss, val_acc, avg_val_confidence

def train_one_epoch_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epoch, total_epochs, verbose=False):
    train_loss, train_acc, epoch_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_avg_confidence = evaluate(model, val_loader, criterion, device)
    if verbose:
        print(f"Epoch {epoch+1}/{total_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Avg Conf: {val_avg_confidence:.4f}, Time: {epoch_time}")
    
    return train_loss, train_acc, val_loss, val_acc, val_avg_confidence

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs, device, scheduler_warmup=None, scheduler_plateau=None, verbose=False):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_avg_confidence": [],
        "learning_rate": []
    }
    final_test_evaluation = {
        "test_accuracy": 0.0,
        "test_loss": 0.0
    }

    if verbose:
        print(f"Starting the training of {model.__class__.__name__}...") 

    for epoch in range(epochs):
        train_loss, train_acc, val_loss, val_acc, val_avg_confidence = train_one_epoch_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, device, epoch, epochs, verbose
        )

        # Step schedulers: warmup for the first warmup_epochs, plateau afterwards
        if scheduler_warmup and epoch < scheduler_warmup.total_iters:
            scheduler_warmup.step()
        elif scheduler_plateau:
            scheduler_plateau.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_avg_confidence"].append(val_avg_confidence)
        history["learning_rate"].append(current_lr)
    
    # Final evaluation on the test set
    test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)
    final_test_evaluation["test_accuracy"] = test_acc
    final_test_evaluation["test_loss"] = test_loss

    return history, final_test_evaluation, model
