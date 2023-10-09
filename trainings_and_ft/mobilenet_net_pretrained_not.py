
from ..utils.training_utils import create_data_loaders, train_model, save_model, init_seed, create_model

if __name__ == "__main__":
    
    init_seed()
    train_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data/train/"
    val_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data/valid/"
    checkpoint_dir = "/home/ray/nfs/autolang_storage/projects/divyam/primary_ft_training/mobilenetv2_net_full_non_pretrained"
    
    batch_size = 256
    num_epochs = 30
    lr = 0.0001
    model_name = "mobilenet_v2" 
    train_loader, val_loader = create_data_loaders(train_dir, val_dir, batch_size)
    model = create_model(model_name, num_classes=525, only_last=False, pretrained=False)
    trained_model = train_model(model, train_loader, val_loader, num_epochs, lr, save_path=checkpoint_dir)
    
    save_model(trained_model, checkpoint_dir)

