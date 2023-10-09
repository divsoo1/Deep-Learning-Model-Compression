import os
import glob
import torch

# Define the root directory where your model folders are located
root_directory = '/home/ray/nfs/autolang_storage/projects/divyam/primary_ft_training'

# Function to find the best validation accuracy in a folder
def find_best_val_acc(folder_path):
    ckpt_files = glob.glob(os.path.join(folder_path, '**', 'training_metrics.ckpt'), recursive=True)
    
    best_acc = 0.0  # Initialize the best accuracy
    
    for ckpt_file in ckpt_files:
        epoch_list, _, _, _, val_acc_list = load_training_metrics(ckpt_file)
        if val_acc_list:
            best_acc = max(best_acc, max(val_acc_list))
    
    return best_acc

# Function to load training metrics from a checkpoint file
def load_training_metrics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    train_accs = []
    val_accs = []
    
    for epoch in checkpoint['epoch_list']:
        train_accs.append(checkpoint['train_acc_list'][epoch].item())
        val_accs.append(checkpoint['val_acc_list'][epoch].item())
    
    return checkpoint['epoch_list'], checkpoint['train_loss_list'], train_accs, checkpoint['val_loss_list'],val_accs

# Loop through all model folders and find the best validation accuracy
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path):
        best_acc = find_best_val_acc(folder_path)
        print(f'Model: {folder_name}, Best Validation Accuracy: {best_acc}')


# epochs, train_loss, train_acc, val_loss, val_acc = load_training_metrics("/home/ray/nfs/autolang_storage/projects/divyam/primary_ft_training/mobilenetv2/training_metrics.ckpt")