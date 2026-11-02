"""Small, notebook-friendly utilities for RecVis Assignment 3."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class NpzImageDataset(Dataset):
    """Read the public NPZ archive format."""

    def __init__(self, path: str | Path, transform=None, require_labels: bool = False):
        with np.load(path) as archive:
            if "images" not in archive:
                raise ValueError(f"{path} has no 'images' array.")
            self.images = archive["images"].astype(np.uint8)
            self.ids = archive["ids"].astype(np.int64) if "ids" in archive else np.arange(len(self.images))
            self.labels = archive["labels"].astype(np.int64) if "labels" in archive else None
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"Expected N x H x W x 3 uint8 images, got {self.images.shape}.")
        if require_labels and self.labels is None:
            raise ValueError(f"{path} has no labels; use it for inference only.")
        if self.labels is not None and len(self.labels) != len(self.images):
            raise ValueError("The image and label counts differ.")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index], mode="RGB")
        if self.transform:
            image = self.transform(image)
        if self.labels is None:
            return image, int(self.ids[index])
        return image, int(self.labels[index])


def make_transforms(image_size: int = 32, pretrained: bool = False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (CIFAR10_MEAN, CIFAR10_STD)
    if image_size == 32:
        train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
        evaluate = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    else:
        train = T.Compose([T.RandomResizedCrop(image_size, scale=(0.8, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
        evaluate = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])
    return train, evaluate


def make_loaders(input_dir: str | Path, image_size=32, pretrained=False, batch_size=128, workers=2):
    input_dir = Path(input_dir)
    train_tf, eval_tf = make_transforms(image_size, pretrained)
    train_set = NpzImageDataset(input_dir / "train.npz", train_tf, require_labels=True)
    val_set = NpzImageDataset(input_dir / "validation.npz", eval_tf, require_labels=True)
    kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return DataLoader(train_set, shuffle=True, **kwargs), DataLoader(val_set, shuffle=False, **kwargs)


def build_model(name="resnet18", pretrained=False, image_size=32):
    """Build the supported supervised notebook baselines."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if image_size == 32 and not pretrained:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        return model
    if name == "vit_b_16":
        if image_size != 224:
            raise ValueError("The notebook ViT-B baseline uses 224px inputs.")
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
        return model
    raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if hasattr(model, "fc") else model.heads
    for parameter in head.parameters():
        parameter.requires_grad = True


def _run_epoch(model, loader, optimizer, device, scaler=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total = 0
    amp = device.type == "cuda"
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def train(model, train_loader, validation_loader, epochs=10, lr=3e-4, weight_decay=1e-4, device=None):
    """Train a model and return epoch metrics; restore the best validation weights."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    best_accuracy, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = _run_epoch(model, validation_loader, None, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"epoch {epoch:02d}: train_acc={row['train_accuracy']:.4f}, val_acc={row['val_accuracy']:.4f}")
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


@torch.inference_mode()
def predict(model, test_archive: str | Path, image_size=32, pretrained=False, batch_size=256, workers=2, device=None):
    """Return an id/label DataFrame for the unlabeled competition test archive."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transform = make_transforms(image_size, pretrained)
    dataset = NpzImageDataset(test_archive, transform=transform, require_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model.to(device).eval()
    ids, labels = [], []
    for images, batch_ids in loader:
        labels.extend(model(images.to(device)).argmax(1).cpu().tolist())
        ids.extend(batch_ids.tolist())
    return pd.DataFrame({"id": ids, "label": labels})


@torch.inference_mode()
def clip_zero_shot(test_archive: str | Path, output_path: str | Path, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", batch_size=128):
    """Optional open-source CLIP baseline."""
    import open_clip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    dataset = NpzImageDataset(test_archive, transform=preprocess, require_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prompts = tokenizer([f"a photo of a {name}." for name in CLASS_NAMES]).to(device)
    model.to(device).eval()
    text_features = model.encode_text(prompts)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    ids, labels = [], []
    for images, batch_ids in loader:
        image_features = model.encode_image(images.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        labels.extend((100 * image_features @ text_features.T).argmax(1).cpu().tolist())
        ids.extend(batch_ids.tolist())
    submission = pd.DataFrame({"id": ids, "label": labels})
    submission.to_csv(output_path, index=False)
    return submission

