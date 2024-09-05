from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tvt
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from icecream import ic
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from typing import Iterable, Tuple

import wandb

from ..src.boe import BattleOfExperts
from ..src.resnet import ResNet
from .get_args import get_args

args = get_args()
DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

if args.debug:
    ic.configureOutput(includeContext=True)
    ic.enable()
    torch.autograd.set_detect_anomaly(True)
else:
    ic.disable()
    torch.autograd.set_detect_anomaly(False)

if not Path("battle-of-experts/weights").exists():
    Path("battle-of-experts/weights").mkdir(parents=True)

loss_function = nn.CrossEntropyLoss()

if "boe" in args.model:
    model = BattleOfExperts(num_classes=100, lr=args.lr)
elif "resnet" in args.model:
    model = ResNet(num_classes=100, model_name=args.model, lr=args.lr)
else:
    raise ValueError(f"Unknown model {args.model}")

if args.ckpt is not None:
    print(f"Loading weights from {args.ckpt}")
    weights = torch.load(args.ckpt)
    print(model.load_state_dict(weights))

model = model.to(DEVICE)

num_params = 0
for p in model.parameters():
    if torch.is_complex(p):
        num_params += p.numel() * 2
    else:
        num_params += p.numel()

print(f"{num_params:,} trainable parameters")

if args.only_print_params:
    raise SystemExit

train = CIFAR100(
    root="./battle-of-experts/data",
    train=True,
    download=True,
    transform=tvt.Compose([tvt.AutoAugment(AutoAugmentPolicy.CIFAR10), tvt.ToTensor()]),
)
test = CIFAR100(
    root="./battle-of-experts/data",
    train=False,
    download=True,
    transform=tvt.ToTensor(),
)

train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
    train,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=4,
)
test_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
    test,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
)

if args.log:
    wandb.init(project="battle-of-experts", config=args, name=args.name)
    include_fn = lambda path: path.endswith(".py")
    wandb.run.log_code("./battle-of-experts", include_fn=include_fn)

global_step = 0
train_metrics = {"train_loss": 0.0, "train_error": 0.0}
test_metrics = {"test_loss": 0.0, "test_error": 0.0}
for epoch in range(args.epochs):

    model.train()
    train_pbar = tqdm(train_loader, leave=False, smoothing=0.01)

    for step, (images, labels) in enumerate(train_pbar):

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, labels = images.to(torch.float32), labels.to(torch.long)

        train_metrics.update(model.step(images, labels, step_type="train"))
        if args.log and step % args.log_interval == 0:
            wandb.log(train_metrics, step=global_step)

        if step % args.log_interval == 0:
            train_pbar.set_description(
                f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:01.4f} "
                f"| Train Err: {train_metrics['train_error']:02.2%} "
                f"| Test Err: {test_metrics['test_error']:02.2%}"
            )

        global_step += 1

    model.eval()

    with torch.no_grad():
        test_pbar = tqdm(test_loader, leave=False, smoothing=0.01)
        test_metrics_list = []
        for images, labels in test_pbar:

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = images.to(torch.float32), labels.to(torch.long)

            test_metrics_list.append(model.step(images, labels, step_type="test"))

    keys = test_metrics_list[0].keys()
    for key in keys:
        test_metrics[key] = sum(m[key] for m in test_metrics_list) / len(
            test_metrics_list
        )

    if args.log:
        wandb.log(test_metrics, step=global_step)

    if args.save:
        torch.save(model.state_dict(), f"battle-of-experts/weights/{epoch:03d}.ckpt")
