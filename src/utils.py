import torch


def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    _, predicted = torch.max(predictions, dim=-1)
    return (predicted == labels).sum().item() / labels.shape[0]
