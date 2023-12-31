from ..utils.training_utils import create_data_loaders, save_model, load_model
from ..utils.quantization_utils import prepare_for_quantization, convert_to_quantized_model, evaluate_quantized_model, get_model_size, evaluate
from ..utils import paths

if __name__ == "__main__":

    checkpoint_dir = paths.RESNET50_QUANTIZABLE_TRAINING_FULL_CHECKPOINTDIR
    quantized_checkpoint_dir = paths.RESNET50_QUANTIZED_CHECKPOINTDIR
    model_name = "resnet_quantizable"
    num_classes = 525
    batch_size = 256
    num_calibration_batches = 10

    train_loader, val_loader, test_loader = create_data_loaders(paths.TRAIN_DIR, paths.VALID_DIR, batch_size, test_dir=paths.TEST_DIR)

    model_fp32 = load_model(model_name, checkpoint_dir, num_classes, only_last=False, pretrained=True)
    model_fp32 = model_fp32.to('cpu')

    num_params_base, size_mb_base = get_model_size(model_fp32)
    inference_time_base, accuracy_base = evaluate(model_fp32, test_loader, 'cpu')
    print("Baseline Inference Time: ", inference_time_base)
    print("Baseline Accuracy: ", accuracy_base, '%')
    print("Baseline Model Size:", size_mb_base, "MBs")
    print("Baseline Number of Parameters:", num_params_base)

    model_fp32_prepared = prepare_for_quantization(model_fp32, num_calibration_batches, train_loader)

    model_quantized = convert_to_quantized_model(model_fp32_prepared)

    num_params_quantized, size_mb_quantized, inference_time_quantized, accuracy_quantized = evaluate_quantized_model(model_quantized, test_loader, 'cpu')
    print("Quantized Inference Time: ", inference_time_quantized)
    print("Quantized Accuracy: ", accuracy_quantized, '%')
    print("Quantized Model Size:", size_mb_quantized, "MBs")
    print("Quantized Number of Parameters:", num_params_quantized)

    save_model(model_quantized, quantized_checkpoint_dir)


