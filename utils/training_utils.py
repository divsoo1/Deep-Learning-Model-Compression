import os
import copy
import torch
import torch.optim as optim
from torch import nn
from torchvision import models
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from ..models.resnet_quantizable import resnet50_quantizable
from transformers import enable_full_determinism


def init_seed():
    # Set the random seed manually for reproducibility.
    
	os.environ['PYTHONHASHSEED'] = str(42)
	enable_full_determinism(42)

def create_data_loaders(train_dir, val_dir,  batch_size, test_dir=None):
    """ Create data loaders for the given train, val and test directories.
    """
    
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = ImageFolder(root=train_dir, transform=transform)
    val_data = ImageFolder(root=val_dir, transform=transform)
    if test_dir:
        test_data = ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    if test_dir:
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader

def create_model(model_name, num_classes, only_last=True, pretrained=True):
    """ Create a model with the given name and number of classes.
    """
    
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=pretrained)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
    elif model_name == 'squeeze_net':
        model = models.squeezenet1_0(pretrained=pretrained)
    elif model_name == "resnet_quantizable":
        model = resnet50_quantizable(pretrained=pretrained)
    else:
        raise ValueError("Unsupported model name")

    full_train = not only_last
    for param in model.parameters():
        param.requires_grad = full_train

    if model_name == "resnet50" or model_name == "resnet101" or model_name == "resnet152" or model_name == "resnet_quantizable":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "mobilenet_v2" :
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "squeeze_net":
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
    elif model_name == "densenet121":
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    return model

def train_model(model, train_loader, val_loader, num_epochs, lr=0.001, save_path="model_checkpoints"):
    """ Train the given model for the given number of epochs.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1: 
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    len_trainset = len(train_loader.dataset)
    len_valset = len(val_loader.dataset)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        running_loss = 0.0
        running_corrects = 0

        trainloader = tqdm(train_loader, total=len(train_loader))

        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            trainloader.set_postfix(loss=running_loss / len_trainset, acc=running_corrects.double() / len_trainset)

        epoch_loss = running_loss / len_trainset
        epoch_acc = running_corrects.double() / len_trainset
        print("Train Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))

        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0

        valloader = tqdm(val_loader, total=len(val_loader))

        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss_val += loss.item() * inputs.size(0)
            running_corrects_val += torch.sum(preds == labels.data)

        epoch_loss_val = running_loss_val / len_valset
        epoch_acc_val = running_corrects_val.double() / len_valset

        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

        print("Val Loss: {:.4f} Acc: {:.4f}".format(epoch_loss_val, epoch_acc_val))
        print()
        print("Best val Acc: {:.4f}".format(best_acc))

        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)
        val_loss_list.append(epoch_loss_val)
        val_acc_list.append(epoch_acc_val)

    checkpoint = {
        'epoch_list': epoch_list,
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'val_loss_list': val_loss_list,
        'val_acc_list': val_acc_list
    }
    os.makedirs(save_path, exist_ok=True)
    checkpoint_file = os.path.join(save_path, "training_metrics.ckpt")
    torch.save(checkpoint, checkpoint_file)

    model.load_state_dict(best_model_wts)
    return model

def save_model(model, checkpoint_dir="model_checkpoints"):
    """ Save the given model to the given checkpoint directory.
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_checkpoint_path)


def load_model(model_name, checkpoint_path, num_classes, only_last, pretrained):
    """ Load the model from the given checkpoint path.
    """
    
    model = create_model(model_name, num_classes, only_last=only_last, pretrained=pretrained)
    checkpoint = torch.load(checkpoint_path)
    if all(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)

    return model


def load_training_metrics(checkpoint_path):
    """ Load the training metrics from the given checkpoint path.
    """
    
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['epoch_list'], checkpoint['train_loss_list'], checkpoint['train_acc_list'], checkpoint['val_loss_list'], checkpoint['val_acc_list']