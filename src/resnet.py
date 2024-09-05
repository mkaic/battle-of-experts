import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .utils import calculate_accuracy


class ResNet(nn.Module):
    def __init__(self, num_classes: int, model_name: str, lr: float):
        super().__init__()
        match model_name:
            case "resnet18":
                self.model = resnet18(num_classes=num_classes)
            case "resnet34":
                self.model = resnet34(num_classes=num_classes)
            case "resnet50":
                self.model = resnet50(num_classes=num_classes)
            case "resnet101":
                self.model = resnet101(num_classes=num_classes)
            case "resnet152":
                self.model = resnet152(num_classes=num_classes)
            case _:
                raise ValueError(f"Unknown model {model_name}")

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def step(self, input_tensor, label_tensor, step_type):
        outputs = self.model(input_tensor)
        loss = self.loss_function(outputs, label_tensor)
        accuracy = calculate_accuracy(outputs, label_tensor)

        match step_type:
            case "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            case "test":
                pass
            case _:
                raise ValueError(f"Unknown step type {step_type}")

        return {f"{step_type}_loss": loss.item(), f"{step_type}_error": 1 - accuracy}
