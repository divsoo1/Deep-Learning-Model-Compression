# Deep Machine Learning Project

Welcome to the repository for the Deep Machine Learning Course (DIT967) project at the University of Gothenburg. This project focuses on implementing various model compression techniques on the Resnet50 architecture and evaluating the tradeoffs of each technique in terms of efficiency, model size, and accuracy.

## Replicating Results

To ensure that the results can be replicated, follow these steps:

1. **Setup Data and Checkpoints Paths**: Modify the `paths.py` file inside the `utils` directory to specify the paths for your data and checkpoints.

2. **Baseline Model Setup**: Run the scripts located within the `training` and `finetuning` directories to establish the baseline models for further experimentation.

3. **Apply Compression Techniques**: Run the scripts located within the `pruning`, `quantization`, and `distillation` directories in any order to apply the model compression techniques and evaluate their impact.

## Utility Functions

All utility functions required for training, fine-tuning, pruning, quantization, and distillation are defined in the `utils` folder. Each technique has its own utility functions script inside directories like `training_utils`, `pruning_utils`, `quantization_utils`, and `distillation_utils`.

## Custom Models

Inside the `models` directory, you can find custom model architectures. This includes one self-defined architecture and a quantizable version of Resnet50, which are used in the project.

## Project Findings

Explore the `findings` folder to view accuracy and loss plots for the trained and fine-tuned models. For a more comprehensive and detailed report on the project, please refer to the project `Model Compression Evaulation`.
 

## References

This project was developed with inspiration from the following sources:
- [Maximizing Model Performance with Knowledge Distillation in PyTorch](https://medium.com/artificialis/maximizing-model-performance-with-knowledge-distillation-in-pytorch-12b3960a486a)
- [Resnet50 Quantization](https://github.com/zanvari/resnet50-quantization)
- [Neural Network Pruning](https://github.com/arjun-majumdar/Neural_Network_Pruning)

Feel free to reach out if you have any questions or need further information about this project.
