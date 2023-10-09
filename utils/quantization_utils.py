import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.quantization as quantization

def evaluate(model, dataloader, device):
    """ Evaluate the model on the given dataloader.
    """
    model.eval()
    total_time, correct = 0, 0
    total_samples = len(dataloader.dataset)
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluating', ncols=100):
            imgs, targets = data
            imgs = imgs.to(device) 
            targets = targets.to(device)  
            
            start = time.time()
            outputs = model(imgs)
            end = time.time()
            delta = end - start
            total_time += delta
            
            _, pred_idx = outputs.max(1)
            correct += (targets == pred_idx).sum().item()
    
    inference_time = total_time / total_samples
    accuracy = (correct / total_samples) * 100
    return inference_time, accuracy


def get_model_size(model):
    """ Returns the number of parameters and the size of the model in MB.
    """
    param_mem = 0
    param_num = 0
    for param in model.parameters():
        param_num += param.nelement()
        param_mem += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_mem + buffer_size) / 1024**2
    return param_num, size_all_mb


def prepare_for_quantization(model, num_calibration_batches, train_loader):
    model.eval()
    model.qconfig = quantization.get_default_qconfig('x86')
    model_prepared = quantization.prepare(model)
    model_prepared.eval()

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if i > num_calibration_batches:
                break
            model_prepared(data[0].to('cpu'))

    for name, module in model_prepared.named_modules():
        if isinstance(module, nn.Conv2d) and "relu" in name:
            quantization.fuse_modules(model_prepared, [name], inplace=True)

    return model_prepared

def convert_to_quantized_model(prepared_model):
    return quantization.convert(prepared_model)

def evaluate_quantized_model(model, test_loader, device):
    num_params_quantized, size_mb_quantized = get_model_size(model)
    inference_time_quantized, accuracy_quantized = evaluate(model, test_loader, device)
    return num_params_quantized, size_mb_quantized, inference_time_quantized, accuracy_quantized


    # # Create a comparison plot
    # plt.figure(figsize=(8, 6))
    # labels = ['Baseline', 'Quantized']
    # accuracy_values = [accuracy_base, accuracy_quantized]
    # size_values = [size_mb_base, size_mb_quantized]
    # plt.bar(labels, accuracy_values, color='blue', alpha=0.7, label='Accuracy')
    # plt.bar(labels, size_values, color='orange', alpha=0.7, label='Model Size (MB)')
    # plt.xlabel('Models')
    # plt.ylabel('Values')
    # plt.title('Baseline vs. Quantized Model Comparison')
    # plt.legend()
    # plt.savefig(quantized_checkpoint_dir.replace(".pt", "_comparison_plot.png"))

    # plt.show()