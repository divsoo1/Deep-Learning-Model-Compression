import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def loss_kd(outputs, labels, teacher_outputs, temperature, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / temperature, dim=1),
                             F.softmax(teacher_outputs / temperature, dim=1)) * (alpha * temperature * temperature) + \
              F.cross_entropy(outputs, labels) * (1.0 - alpha)
    return KD_loss

def get_outputs(model, dataloader, device):
    '''
    Used to get the output of the teacher network
    '''
    model.eval()
    outputs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs_batch, labels_batch = inputs.to(device), labels.to(device)
            output_batch = model(inputs_batch)
            # Ensure the output is on the same device as specified by 'device'
            output_batch = output_batch.detach().cpu().numpy() if device == 'cuda' else output_batch.detach().numpy()
            outputs.append(output_batch)

    return outputs


def train_kd(model, teacher_out, optimizer, loss_kd, dataloader, temperature, alpha, device):
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
    # Remove the check for multiple GPUs and DataParallel
    # Replace device with 'cuda' or 'cpu'
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

    # Lists to store epoch, loss, and accuracy values
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Training the student with the soft labels as the outputs from the teacher
        # and using the loss_kd function
        train_loss, train_acc = train_kd(model, outputs_teacher_train, optimizer, loss_kd, trainloader, temperature, alpha, device)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))

        # Evaluating the student network
        val_loss, val_acc = eval_kd(model, outputs_teacher_val, optimizer, loss_kd, valloader, temperature, alpha, device)
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

        # Append epoch, loss, and accuracy values to lists
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc.item())
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Best val Acc: {:.4f}'.format(best_acc))

            # Save the best model's state dictionary to a file
            if save_path is not None:
                checkpoint_path = os.path.join(save_path, f"best_model_epoch{epoch}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print('Saved the best model to:', checkpoint_path)

    model.load_state_dict(best_model_wts)

    # Save the training and validation metrics to a checkpoint file
    if save_path is not None:
        checkpoint_file = os.path.join(save_path, "training_metrics.ckpt")
        torch.save({
            'epoch_list': epoch_list,
            'train_loss_list': train_loss_list,
            'train_acc_list': train_acc_list,
            'val_loss_list': val_loss_list,
            'val_acc_list': val_acc_list
        }, checkpoint_file)

    # Optionally return the path to the saved best model
    return model