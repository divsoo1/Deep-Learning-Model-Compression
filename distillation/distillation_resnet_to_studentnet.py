import torch.optim as optim
from ..utils.training_utils import create_data_loaders, save_model, init_seed, load_model
from ..utils.distillation_utils import train_and_evaluate_kd, loss_kd
from ..models.student_model import StudentNet
from ..utils import paths


if __name__ == "__main__":

    init_seed()
    batch_size = 256
    lr = 0.001
    teacher_model_name = "resnet50"
    tacher_model_checkpoints = paths.RESNET50_FULL_TRAINING
    temperature = 1
    alpha = 0.5
    num_classes = 525
    distillation_epochs = 30
    final_checkpoint_dir = paths.STUDENTNET_DISTILLATION_CHECKPOINTDIR

    train_loader, val_loader = create_data_loaders(paths.TRAIN_DIR, paths.VALID_DIR, batch_size
                                                )
    teacher_model = load_model(teacher_model_name, tacher_model_checkpoints, num_classes=num_classes, only_last=True, pretrained=True)
    student_model = StudentNet()
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    distilled_model = train_and_evaluate_kd(
        student_model, teacher_model, optimizer, loss_kd, train_loader, val_loader, 
        temperature=temperature, alpha=alpha, num_epochs=distillation_epochs, save_path=final_checkpoint_dir
    )

    save_model(distilled_model, checkpoint_dir=final_checkpoint_dir)




