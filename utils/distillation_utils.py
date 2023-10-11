import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def loss_kd(outputs, labels, teacher_outputs, temperature, alpha):
    """
    Calculate the knowledge distillation loss between student and teacher networks.

    Parameters:
        outputs (Tensor): Predicted class probabilities by the student network.
        labels (Tensor): True class labels.
        teacher_outputs (Tensor): Predicted class probabilities by the teacher network.
        temperature (float): Temperature parameter for softening the distributions.
        alpha (float): Weighting factor for the knowledge distillation loss.

    Returns:
        KD_loss (Tensor): Knowledge distillation loss.
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / temperature, dim=1),
                             F.softmax(teacher_outputs / temperature, dim=1)) * (alpha * temperature * temperature) + \
              F.cross_entropy(outputs, labels) * (1.0 - alpha)
    return KD_loss

def get_outputs(model, dataloader, device):
    """
    Get the output of a neural network for a given dataset.

    Parameters:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Data loader for the dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        outputs (list): List of model outputs for each batch in the dataset.
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs_batch, labels_batch = inputs.to(device), labels.to(device)
            output_batch = model(inputs_batch)
            output_batch = output_batch.detach().cpu().numpy() if device == 'cuda' else output_batch.detach().numpy()
            outputs.append(output_batch)

    return outputs

def train_kd(model, teacher_out, optimizer, loss_kd, dataloader, temperature, alpha, device):
    """
    Train a student network using knowledge distillation.

    Parameters:
        model (nn.Module): The student network.
        teacher_out (list): Teacher network outputs for the training dataset.
        optimizer (Optimizer): The optimizer for training.
        loss_kd (function): Knowledge distillation loss function.
        dataloader (DataLoader): Data loader for the training dataset.
        temperature (float): Temperature parameter for softening the distributions.
        alpha (float): Weighting factor for the knowledge distillation loss.
        device (str): Device to run the training on ('cuda' or 'cpu').

    Returns:
        epoch_loss (float): Average training loss for the epoch.
        epoch_acc (float): Training accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for i, (images, labels) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Training")):
        inputs = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs_teacher = torch.from_numpy(teacher_out[i]).to(device)
        loss = loss_kd(outputs, labels, outputs_teacher, temperature, alpha)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def eval_kd(model, teacher_out, optimizer, loss_kd, dataloader, temperature, alpha, device):
    """
    Evaluate a student network using knowledge distillation.

    Parameters:
        model (nn.Module): The student network.
        teacher_out (list): Teacher network outputs for the validation dataset.
        optimizer (Optimizer): The optimizer for evaluation.
        loss_kd (function): Knowledge distillation loss function.
        dataloader (DataLoader): Data loader for the validation dataset.
        temperature (float): Temperature parameter for softening the distributions.
        alpha (float): Weighting factor for the knowledge distillation loss.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        epoch_loss (float): Average validation loss for the epoch.
        epoch_acc (float): Validation accuracy for the epoch.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for i, (images, labels) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Validation")):
        inputs = images.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs_teacher = torch.from_numpy(teacher_out[i]).to(device)
        loss = loss_kd(outputs, labels, outputs_teacher, temperature, alpha)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def train_and_evaluate_kd(model, teacher_model, optimizer, loss_kd, trainloader, valloader, temperature, alpha, num_epochs=25, save_path=None, device="cuda"):
    """
    Train and evaluate a student network using knowledge distillation.

    Parameters:
        model (nn.Module): The student network.
        teacher_model (nn.Module): The teacher network.
        optimizer (Optimizer): The optimizer for training and evaluation.
        loss_kd (function): Knowledge distillation loss function.
        trainloader (DataLoader): Data loader for the training dataset.
        valloader (DataLoader): Data loader for the validation dataset.
        temperature (float): Temperature parameter for softening the distributions.
        alpha (float): Weighting factor for the knowledge distillation loss.
        num_epochs (int): Number of training epochs (default: 25).
        save_path (str): Path to save the best model checkpoint (default: None).
        device (str): Device to run the training and evaluation on ('cuda' or 'cpu').

    Returns:
        model (nn.Module): The trained student model.
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA is not available. Switching to CPU.")
            device = 'cpu'
        else:
            print("Using CUDA.")
            model = model.to(device)
            teacher_model = teacher_model.to(device)
    else:
        device = 'cpu'
        print("Using CPU.")

    os.makedirs(save_path, exist_ok=True)
    teacher_model.eval()
    best_model_wts = copy.deepcopy(model.state_dict())
    outputs_teacher_train = get_outputs(teacher_model, trainloader, device)
    outputs_teacher_val = get_outputs(teacher_model, valloader, device)
    print("Teacher’s outputs are computed. Starting the training process...")
    best_acc = 0.0

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss, train_acc = train_kd(model, outputs_teacher_train, optimizer, loss_kd, trainloader, temperature, alpha, device)
        print('Train Loss: {:.4f} Acc: {:.4f'.format(train_loss, train_acc))

        val_loss, val_acc = eval_kd(model, outputs_teacher_val, optimizer, loss_kd, valloader, temperature, alpha, device)
        print('Val Loss: {:.4f} Acc: {:.4f'.format(val_loss, val_acc))

        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc.item())
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Best val Acc: {:.4f}'.format(best_acc))

            if save_path is not None:
                checkpoint_path = os.path.join(save_path, f"best_model_epoch{epoch}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print('Saved the best model to:', checkpoint_path)

    model.load_state_dict(best_model_wts)

    if save_path is not None:
        checkpoint_file = os.path.join(save_path, "training_metrics.ckpt")
        torch.save({
            'epoch_list': epoch_list,
            'train_loss_list': train_loss_list,
            'train_acc_list': train_acc_list,
            'val_loss_list': val_loss_list,
            'val_acc_list': val_acc_list
        }, checkpoint_file)

    return model
