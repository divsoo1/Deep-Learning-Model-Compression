from ..utils.training_utils import create_data_loaders, train_model, save_model, init_seed, create_model

if __name__ == "__main__":
    
    init_seed()
    train_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data/train/"
    val_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data/test/"
    checkpoint_dir = "/home/ray/nfs/autolang_storage/projects/divyam/primary_ft_training/resnet50"
    
    batch_size = 256
    num_epochs = 15
    lr = 0.001
    model_name = "resnet50" 
    train_loader, val_loader = create_data_loaders(train_dir, val_dir, batch_size)
    model = create_model(model_name, num_classes=525, only_last=True)
    trained_model = train_model(model, train_loader, val_loader, num_epochs, lr, save_path=checkpoint_dir)
    
    save_model(trained_model, checkpoint_dir)

