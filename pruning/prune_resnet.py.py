
import torch
import copy

from ..utils.training_utils import create_data_loaders, train_model, save_model, init_seed, create_model
from ..utils.pruning_utils import evaluate_model, measure_global_sparsity, iterative_pruning_finetuning, remove_parameters
from ..utils.quantization_utils import get_model_size

if __name__ == "__main__":

    init_seed()
    train_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data/train/"
    val_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data/valid/"
    test_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data/test/"
    checkpoint_dir = "/home/ray/nfs/autolang_storage/projects/divyam/pruning/unpruned_resnet50_trained"

    model_name = "resnet50"
    num_classes = 525
    batch_size = 64
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_data_loaders(train_dir, val_dir, batch_size, test_dir=test_dir)

    trained_model = create_model(model_name, num_classes, only_last=False, pretrained=True)
    trained_model.conv1 = torch.nn.Conv2d(
        in_channels=3, out_channels=64,
        kernel_size=(3, 3), stride=(1, 1),
        padding=(1, 1), bias=False
    )

    state_dict = torch.load("/home/ray/nfs/autolang_storage/projects/divyam/pruning/unpruned_resnet50_trained/final_model.pt")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    trained_model.load_state_dict(state_dict)
    trained_model.to(device)

    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    learning_rate = 0.01
    learning_rate_decay = 1

    _, eval_accuracy = evaluate_model(
        model=trained_model, test_loader=test_loader,
        device=device, criterion=None)

    num_zeros, num_elements, sparsity = measure_global_sparsity(trained_model)
    print(f"Global sparsity = {sparsity:.3f}% & val_accuracy = {eval_accuracy * 100:.3f}%")

    print("Iterative Pruning + Fine-Tuning...")

    pruned_model = copy.deepcopy(trained_model)

    pruned_model = iterative_pruning_finetuning(
        model=pruned_model, train_loader=train_loader,
        test_loader=test_loader, device=device,
        learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
        l1_regularization_strength=l1_regularization_strength, l2_regularization_strength=l2_regularization_strength,
        conv2d_prune_amount=0.2, linear_prune_amount=0.1,
        num_iterations=5, num_epochs_per_iteration=5,
        model_filename_prefix="pruned_model", model_dir="/home/ray/nfs/autolang_storage/projects/divyam/pruning/saved_models",
        grouped_pruning=True)

    remove_parameters(model=pruned_model)

    _, eval_accuracy = evaluate_model(
        model=pruned_model, test_loader=test_loader,
        device=device, criterion=None)

    num_zeros, num_elements, sparsity = measure_global_sparsity(pruned_model)

    print(f"Global sparsity = {sparsity:.3f} & val_accuracy = {eval_accuracy:.3f}")
    save_model(pruned_model, "/home/ray/nfs/autolang_storage/projects/divyam/pruning/pruned_model_checkpoints")

    param_num, size_all_mb = get_model_size(pruned_model)


