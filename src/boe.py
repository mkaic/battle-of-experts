import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .utils import calculate_accuracy


class BattleOfExperts(nn.Module):
    def __init__(
        self, num_classes: int, lr: float, num_experts: int = 2, hidden_dim: int = 512
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.experts = nn.ModuleList()
        self.expert_optimizers = []
        for _ in range(num_experts):
            self.experts.append(resnet18(num_classes=hidden_dim * 2))

        self.decider = resnet18(num_classes=hidden_dim * 2)

        self.linear_out = nn.Linear(hidden_dim, num_classes)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def step(self, input_tensor, label_tensor, step_type):
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(input_tensor))
        expert_outputs = torch.stack(
            expert_outputs, dim=1
        )  # (batch_size, num_experts, hidden_dim)
        keys = expert_outputs[..., : self.hidden_dim]
        values = expert_outputs[..., self.hidden_dim :]
        queries = self.decider(input_tensor)  # (batch_size, hidden_dim)

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
