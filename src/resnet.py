import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "resnet18"):
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

    def forward(self, input_tensor, label_tensor):
        outputs = self.model(input_tensor)
        loss = loss_function(outputs, labels)

        _, predicted = torch.max(predictions, dim=-1)

        if step > len(train_loader) * 0.9:
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        losses.append(loss.item())
        loss.backward()
