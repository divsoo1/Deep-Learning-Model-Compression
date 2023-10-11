from ..utils.training_utils import create_data_loaders, train_model, save_model, init_seed, create_model
from ..utils import paths

if __name__ == "__main__":
    
    init_seed()
    checkpoint_dir = paths.RESNET50_QUANTIZABLE_TRAINING_FULL_CHECKPOINTDIR
    batch_size = 256
    num_epochs = 20
    lr = 0.0001
    model_name = "resnet_quantizable"
    train_loader, val_loader = create_data_loaders(paths.TRAIN_DIR, paths.VALID_DIR, batch_size)
    model = create_model(model_name, num_classes = 525, only_last= False)
    trained_model = train_model(model, train_loader, val_loader, num_epochs, lr, save_path=checkpoint_dir)
    
    save_model(trained_model, checkpoint_dir)

