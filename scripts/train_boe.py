from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as tvt
import wandb
from icecream import ic
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18
from tqdm import tqdm

ic.configureOutput(includeContext=True)

from ..src.boe import BattleOfExperts
from .get_args import get_args

args = get_args()

DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

if args.debug:
    ic.enable()
    torch.autograd.set_detect_anomaly(True)
else:
    ic.disable()
    torch.autograd.set_detect_anomaly(False)

if not Path("battle-of-experts/weights").exists():
    Path("battle-of-experts/weights").mkdir(parents=True)

loss_function = nn.CrossEntropyLoss()

match args.model:
    case "boe":
        model = BattleOfExperts(num_classes=100, input_channels=3)
        optimizers = model.get_optimizers(lr=args.lr)
    case "resnet":
        model = resnet18(num_classes=100)
        optimizers = [AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)]

if args.ckpt is not None:
    print(f"Loading weights from {args.ckpt}")
    weights = torch.load(args.ckpt)
    weights.pop("pos_enc")
    print(model.load_state_dict(weights))

model = model.to(DEVICE)

if args.print_params:
    print(model)

num_params = 0
for p in model.parameters():
    if torch.is_complex(p):
        num_params += p.numel() * 2
    else:
        num_params += p.numel()

print(f"{num_params:,} trainable parameters")

if not args.print_params:

    if args.logs:
        wandb.init(project="battle-of-experts", config=args, name=args.name)
        include_fn = lambda path: path.endswith(".py")
        wandb.run.log_code("./battle-of-experts", include_fn=include_fn)
        wandb.watch(model, log="parameters", log_freq=390)

    train = CIFAR100(
        root="./battle-of-experts/data",
        train=True,
        download=True,
    )
    test = CIFAR100(
        root="./battle-of-experts/data",
        train=False,
        download=True,
        transform=tvt.ToTensor(),
    )

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    train_accuracy = 0
    test_accuracy = 0
    for epoch in range(args.epochs):

        model.train()
        pbar = tqdm(train_loader, leave=False)

        total = 0
        correct = 0
        losses = []
        for step, (images, labels) in enumerate(pbar):

            for o in optimizers:
                o.zero_grad()

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = images.to(torch.float32), labels.to(torch.long)

            predictions = model(images)

            _, predicted = torch.max(predictions, dim=-1)

            if step > len(train_loader) * 0.9:
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

            loss = loss_function(predictions, labels)

            losses.append(loss.item())
            loss.backward()

            for o in optimizers:
                o.step()

            pbar.set_description(
                f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Train Err: {1 - train_accuracy:.2%} | Test Err: {1 - test_accuracy:.2%}"
            )

        train_accuracy = correct / total

        model.eval()
        if args.save:
            torch.save(
                model.state_dict(), f"battle-of-experts/weights/{epoch:03d}.ckpt"
            )

        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, leave=False):

                images: torch.Tensor
                labels: torch.Tensor

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                images, labels = images.to(torch.float32), labels.to(torch.long)

                predictions = model(images)
                _, predicted = torch.max(predictions, dim=1)

                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total

        if args.logs:
            wandb.log(
                {
                    "train_loss": torch.tensor(losses).mean(),
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                }
            )
