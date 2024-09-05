import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .utils import calculate_accuracy

from icecream import ic


class BattleOfExperts(nn.Module):
    def __init__(
        self, num_classes: int, lr: float, num_experts: int = 4, hidden_dim: int = 512
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.experts = nn.ModuleList()
        self.expert_keys = nn.ModuleList()
        self.expert_values = nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(resnet18(num_classes=hidden_dim))
            self.expert_keys.append(nn.Linear(hidden_dim, hidden_dim))
            self.expert_values.append(nn.Linear(hidden_dim, hidden_dim))

        self.decider = resnet18(num_classes=hidden_dim)

        self.linear_out = nn.Linear(hidden_dim, num_classes)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def step(self, input_tensor, label_tensor, step_type):

        keys = []
        values = []
        for expert, key_layer, value_layer in zip(
            self.experts, self.expert_keys, self.expert_values
        ):
            logits = F.normalize(expert(input_tensor))
            keys.append(F.normalize(key_layer(logits)))
            values.append(F.normalize(value_layer(logits)))

        keys = torch.stack(keys, dim=1)  # (batch_size, num_experts, hidden_dim)
        values = torch.stack(values, dim=1)  # (batch_size, num_experts, hidden_dim)

        queries = F.normalize(self.decider(input_tensor)).unsqueeze(
            1
        )  # (batch_size, 1, hidden_dim)

        dot_products = F.normalize(
            (queries * keys).sum(-1)
        )  # (batch_size, num_experts)

        attention = torch.softmax(dot_products, dim=-1).unsqueeze(
            -1
        )  # (batch_size, num_experts, 1)

        outputs = (attention * values).sum(1)  # (batch_size, hidden_dim)
        outputs = self.linear_out(outputs)

        accuracy_loss = self.loss_function(outputs, label_tensor)
        diversity_loss = -1 * torch.mean(torch.std(F.normalize(values), dim=1))

        loss = accuracy_loss + diversity_loss

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
