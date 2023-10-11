import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.utils.prune as prune

def evaluate_model(model, test_loader, device, criterion=None):
    """
    Evaluate a neural network model on a test dataset.

    Parameters:
        model (nn.Module): The neural network model to evaluate.
        test_loader (DataLoader): Data loader for the test dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
        criterion (nn.Module, optional): Loss criterion for evaluation (default: None).

    Returns:
        eval_loss (float): Average evaluation loss.
        eval_accuracy (float): Evaluation accuracy.
    """
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def create_classification_report(model, device, test_loader):
    """
    Create a classification report for a neural network model.

    Parameters:
        model (nn.Module): The neural network model.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
        test_loader (DataLoader): Data loader for the test dataset.

    Returns:
        classification_report (str): Classification report as a string.
    """
    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            y_true += data[1].numpy().tolist()
            images, _ = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.cpu().numpy().tolist()

    classification_report = metrics.classification_report(
        y_true=y_true, y_pred=y_pred)

    return classification_report

def remove_parameters(model):
    """
    Remove pruning from model parameters.

    Parameters:
        model (nn.Module): The neural network model with pruned parameters.

    Returns:
        model (nn.Module): The model with pruning removed.
    """
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def compute_final_pruning_rate(pruning_rate, num_iterations):
    """
    Compute the final pruning rate for iterative pruning.

    Parameters:
        pruning_rate (float): Pruning rate for each iteration.
        num_iterations (int): Number of pruning iterations.

    Returns:
        final_pruning_rate (float): Final pruning rate.
    """
    final_pruning_rate = 1 - (1 - pruning_rate) ** num_iterations
    return final_pruning_rate

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    """
    Measure sparsity of a neural network module.

    Parameters:
        module (nn.Module): The module to measure sparsity for.
        weight (bool): Measure weight sparsity (default: True).
        bias (bool): Measure bias sparsity (default: False).
        use_mask (bool): Use masking for measurement (default: False).

    Returns:
        num_zeros (int): Number of zero elements.
        num_elements (int): Total number of elements.
        sparsity (float): Sparsity ratio.
    """
    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(
    model, weight=True, bias=False, conv2d_use_mask=False, linear_use_mask=False):
    """
    Measure the global sparsity of a neural network model.

    Parameters:
        model (nn.Module): The neural network model.
        weight (bool): Measure weight sparsity (default: True).
        bias (bool): Measure bias sparsity (default: False).
        conv2d_use_mask (bool): Use masking for Conv2D layers (default: False).
        linear_use_mask (bool): Use masking for Linear layers (default: False).

    Returns:
        num_zeros (int): Number of zero elements.
        num_elements (int): Total number of elements.
        sparsity (float): Global sparsity ratio.
    """
    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
        elif isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def fine_tune_train_model(
    model, train_loader, test_loader, device,
    l1_regularization_strength=0, l2_regularization_strength=1e-4,
    learning_rate=1e-1, num_epochs=20):
    """
    Fine-tune a neural network model on a training dataset.

    Parameters:
        model (nn.Module): The neural network model to fine-tune.
        train_loader (DataLoader): Data loader for the training dataset.
        test_loader (DataLoader): Data loader for the test dataset.
        device (str): Device to run the fine-tuning on ('cuda' or 'cpu').
        l1_regularization_strength (float, optional): L1 regularization strength (default: 0).
        l2_regularization_strength (float, optional): L2 regularization strength (default: 1e-4).
        learning_rate (float, optional): Learning rate (default: 1e-1).
        num_epochs (int, optional): Number of fine-tuning epochs (default: 20).

    Returns:
        fine_tuned_model (nn.Module): The fine-tuned model.
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate,
        momentum=0.9, weight_decay=l2_regularization_strength
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 15],
        gamma=0.1, last_epoch=-1)
    
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(
        model=model, test_loader=test_loader,
        device=device, criterion=criterion)
    
    print(f"Pre fine-tuning: val_loss = {eval_loss:.3f} & val_accuracy = {eval_accuracy * 100:.3f}%")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            l1_reg = torch.tensor(0.).to(device)
            for module in model.modules():
                mask = None
                weight = None
                for name, buffer in module.named_buffers():
                    if name == "weight_mask":
                        mask = buffer
                for name, param in module.named_parameters():
                    if name == "weight_orig":
                        weight = param
                if mask is not None and weight is not None:
                    l1_reg += torch.norm(mask * weight, 1)

            loss += l1_regularization_strength * l1_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(
            model=model, test_loader=test_loader,
            device=device, criterion=criterion)
        scheduler.step()
        print(f"epoch = {epoch + 1} loss = {train_loss:.3f}, accuracy = {train_accuracy * 100:.3f}%, val_loss = {eval_loss:.3f}, val_accuracy = {eval_accuracy * 100:.3f}% & LR: {optimizer.param_groups[0]['lr']:.4f}")

    return model

def iterative_pruning_finetuning(
    model, train_loader, test_loader, device, learning_rate, l1_regularization_strength,
    l2_regularization_strength, learning_rate_decay=0.1, conv2d_prune_amount=0.2,
    linear_prune_amount=0.1, num_iterations=10, num_epochs_per_iteration=10,
    model_filename_prefix="pruned_model", model_dir="saved_models", grouped_pruning=False):
    """
    Perform iterative pruning and fine-tuning of a neural network model.

    Parameters:
        model (nn.Module): The neural network model to prune and fine-tune.
        train_loader (DataLoader): Data loader for the training dataset.
        test_loader (DataLoader): Data loader for the test dataset.
        device (str): Device to run the pruning and fine-tuning on ('cuda' or 'cpu').
        learning_rate (float): Learning rate for fine-tuning.
        l1_regularization_strength (float): L1 regularization strength.
        l2_regularization_strength (float): L2 regularization strength.
        learning_rate_decay (float, optional): Learning rate decay factor (default: 0.1).
        conv2d_prune_amount (float): Pruning rate for Conv2D layers.
        linear_prune_amount (float): Pruning rate for Linear layers.
        num_iterations (int): Number of pruning iterations.
        num_epochs_per_iteration (int): Number of fine-tuning epochs per iteration.
        model_filename_prefix (str, optional): Prefix for saved model filenames (default: "pruned_model").
        model_dir (str, optional): Directory for saving models (default: "saved_models").
        grouped_pruning (bool, optional): Use grouped (global) pruning (default: False).

    Returns:
        model (nn.Module): The pruned and fine-tuned model.
    """
    for i in range(num_iterations):
        print("\nPruning and Fine-tuning {}/{}".format(i + 1, num_iterations))
        print("Pruning...")

        if grouped_pruning == True:
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
                elif isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(
                        module, name="weight",
                        amount=conv2d_prune_amount)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(
                        module, name="weight",
                        amount=linear_prune_amount)

        _, eval_accuracy = evaluate_model(
            model=model, test_loader=test_loader,
            device=device, criterion=None)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model, weight=True,
            bias=False, conv2d_use_mask=True,
            linear_use_mask=False)

        print(f"Global sparsity = {sparsity * 100:.3f}% & val_accuracy = {eval_accuracy * 100:.3f%}")

        print("\nFine-tuning...")

        fine_tuned_model = fine_tune_train_model(
            model=model, train_loader=train_loader,
            test_loader=test_loader, device=device,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            learning_rate=learning_rate,
            num_epochs=num_epochs_per_iteration)

        _, eval_accuracy = evaluate_model(
            model=fine_tuned_model, test_loader=test_loader,
            device=device, criterion=None)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            fine_tuned_model, weight=True,
            bias=False, conv2d_use_mask=True,
            linear_use_mask=False)

        print(f"Post fine-tuning: Global sparsity = {sparsity * 100:.3f}% & val_accuracy = {eval_accuracy * 100:.3f}%")

    return fine_tuned_model
