import os
import matplotlib.pyplot as plt
import torch

# Define the root directory
root_directory = '/home/ray/nfs/autolang_storage/projects/divyam/primary_ft_training'
plot_directory = '/home/ray/nfs/autolang_storage/projects/divyam/primary_ft_training/plots'

# Function to load and plot training metrics
def load_training_metrics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    train_accs = []
    val_accs = []
    
    for epoch in checkpoint['epoch_list']:
        train_accs.append(checkpoint['train_acc_list'][epoch].item())
        val_accs.append(checkpoint['val_acc_list'][epoch].item())
    
    return checkpoint['epoch_list'], checkpoint['train_loss_list'], train_accs, checkpoint['val_loss_list'], val_accs

# Function to plot metrics
def plot_metrics(epoch_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list, model_folder):
    # Move GPU tensors to CPU if needed

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_loss_list, label='Train Loss')
    plt.plot(epoch_list, val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_folder}')
    plt.legend()
    loss_plot_path = os.path.join(plot_directory, f'{model_folder}_loss_plot.png')
    plt.savefig(loss_plot_path)  # Save the plot as an image file
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_acc_list, label='Train Accuracy')
    plt.plot(epoch_list, val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy - {model_folder}')
    plt.legend()
    accuracy_plot_path = os.path.join(plot_directory, f'{model_folder}_accuracy_plot.png')
    plt.savefig(accuracy_plot_path)  # Save the plot as an image file
    plt.close()

# Loop through all subdirectories in the root directory
for root, dirs, files in os.walk(root_directory):
    for dir_name in dirs:
        metrics_file_path = os.path.join(root, dir_name, 'training_metrics.ckpt')

        # Check if the file exists and is a file (not a directory)
        if os.path.isfile(metrics_file_path):
            print(f'Loading and plotting metrics for folder: {os.path.join(root, dir_name)}')
            epoch_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list = load_training_metrics(metrics_file_path)
            plot_metrics(epoch_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list, dir_name)
