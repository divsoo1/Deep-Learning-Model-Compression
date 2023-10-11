import torch
from ..utils.training_utils import create_data_loaders, save_model, create_model,  train_model, init_seed
import torch.quantization
from ..utils import paths


if __name__== "__main__":

    init_seed()
    checkpoint_dir = paths.RESNET50_PRUNABLE_TRAINING_FULL_CHECKPOINTDIR
    model_name = "resnet50"
    num_classes = 525
    batch_size = 256
    epochs =  20
    train_loader, val_loader, test_loader = create_data_loaders(paths.TRAIN_DIR, paths.VALID_DIR, batch_size, test_dir=paths.TRAIN_DIR)
    model = create_model(model_name, num_classes, only_last=False, pretrained=True)
    model.conv1 = torch.nn.Conv2d(
        in_channels = 3, out_channels = 64,
        kernel_size = (3, 3), stride = (1, 1),
        padding = (1, 1), bias = False
    )
    trained_model = train_model(model, train_loader, val_loader, num_epochs = 20, lr=0.0001, save_path=checkpoint_dir)
    save_model(trained_model, checkpoint_dir)
