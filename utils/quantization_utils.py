import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.quantization as quantization

def evaluate(model, dataloader, device):
    """ Evaluate the model on the given dataloader.
    
    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): Data loader for evaluation.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        inference_time (float): Average inference time per sample.
        accuracy (float): Evaluation accuracy.
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
    
    Args:
        model (nn.Module): The neural network model.

    Returns:
        param_num (int): Number of parameters in the model.
        size_all_mb (float): Model size in MB.
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
    """ Prepare a model for quantization by calibrating it with a subset of the training data.

    Args:
        model (nn.Module): The neural network model to prepare for quantization.
        num_calibration_batches (int): Number of calibration batches.
        train_loader (DataLoader): Data loader for training data.

    Returns:
        model_prepared (nn.Module): The prepared model.
    """
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
    """ Convert a prepared model to a quantized model.

    Args:
        prepared_model (nn.Module): The prepared model.

    Returns:
        model_quantized (nn.Module): The quantized model.
    """
    return quantization.convert(prepared_model)

def evaluate_quantized_model(model, test_loader, device):
    """ Evaluate a quantized model.

    Args:
        model (nn.Module): The quantized model to evaluate.
        test_loader (DataLoader): Data loader for the test dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        num_params_quantized (int): Number of parameters in the quantized model.
        size_mb_quantized (float): Model size in MB for the quantized model.
        inference_time_quantized (float): Average inference time per sample for the quantized model.
        accuracy_quantized (float): Evaluation accuracy for the quantized model.
    """
    num_params_quantized, size_mb_quantized = get_model_size(model)
    inference_time_quantized, accuracy_quantized = evaluate(model, test_loader, device)
    return num_params_quantized, size_mb_quantized, inference_time_quantized, accuracy_quantized
