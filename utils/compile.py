import os
import json
import torch


def load_training_metrics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    train_accs = []
    val_accs = []
    
    for epoch in checkpoint['epoch_list']:
        try:
            train_accs.append(checkpoint['train_acc_list'][epoch].item())
            val_accs.append(checkpoint['val_acc_list'][epoch].item())
        except:
            train_accs.append(checkpoint['train_acc_list'][epoch])
            val_accs.append(checkpoint['val_acc_list'][epoch])
    
    return checkpoint['epoch_list'], checkpoint['train_loss_list'], train_accs, checkpoint['val_loss_list'],val_accs


def find_training_metrics_files(root_dir):
    results = {}
    for root, _, files in os.walk(root_dir):
        if "training_metrics.ckpt" in files:
            folder_name = os.path.basename(root)
            checkpoint_path = os.path.join(root, "training_metrics.ckpt")
            
            # Load training metrics using your function
            epoch_list, train_loss_list, train_accs, val_loss_list, val_accs = load_training_metrics(checkpoint_path)
            
            results[folder_name] = {
                'checkpoint_path': checkpoint_path,
                'epoch_list': epoch_list,
                'train_loss_list': train_loss_list,
                'train_accs': train_accs,
                'val_loss_list': val_loss_list,
                'val_accs': val_accs
            }
    return results

def create_jsonl_file(data, output_file):
    with open(output_file, 'w') as file:
        for folder_name, metrics_data in data.items():
            json.dump({folder_name: metrics_data}, file)
            file.write('\n')

if __name__ == "__main__":
    root_directory = "/home/ray/nfs/autolang_storage/projects/divyam/distillation_training"
    output_jsonl_file = "/home/ray/nfs/autolang_storage/projects/divyam/distillation_metrics.jsonl"

    metrics_data = find_training_metrics_files(root_directory)
    create_jsonl_file(metrics_data, output_jsonl_file)
    print(f"JSONL file '{output_jsonl_file}' created with metrics data.")
