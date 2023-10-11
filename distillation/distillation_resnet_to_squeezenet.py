
import torch.optim as optim
from ..utils.training_utils import create_data_loaders, save_model, init_seed, create_model, load_model
from ..utils.distillation_utils import train_and_evaluate_kd, loss_kd
from ..utils import paths


if __name__ == "__main__":

    init_seed()

    BATCH_SIZE = 256
    LR = 0.001
    TEACHER_MODEL_NAME = "resnet50"
    TEACHER_MODEL_CHECKPOINTS = paths.RESNET50_FULL_TRAINING
    TEMPERATURE = 1
    ALPHA = 0.5
    NUM_CLASSES = 525
    NUM_DISTILLATION_EPOCHS = 30
    FINAL_CHECKPOINT_DIR = paths.SQUEEZENET_DISTILLATION_CHECKPOINTDIR

    train_loader, val_loader = create_data_loaders(paths.TRAIN_DIR, paths.VALID_DIR, BATCH_SIZE)
    teacher_model = load_model(TEACHER_MODEL_NAME, TEACHER_MODEL_CHECKPOINTS, num_classes=NUM_CLASSES, only_last=True, pretrained=True)
    student_model = create_model("squeeze_net", num_classes=NUM_CLASSES, only_last=False, pretrained=False)

    optimizer = optim.Adam(student_model.parameters(), lr=LR)
    distilled_model = train_and_evaluate_kd(
        student_model, teacher_model, optimizer, loss_kd, train_loader, val_loader, 
        temperature=TEMPERATURE, alpha=ALPHA, num_epochs=NUM_DISTILLATION_EPOCHS, save_path=FINAL_CHECKPOINT_DIR
    )

    save_model(distilled_model, checkpoint_dir=FINAL_CHECKPOINT_DIR)




