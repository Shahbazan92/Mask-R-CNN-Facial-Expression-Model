"""
Train a ResNet18 facial-expression classifier.

Supported dataset layouts:
1) Facial Expression DataSet/
      train/<class_name>/*.jpg
      test/<class_name>/*.jpg

2) Facial Expression DataSet/
      train/train/<class_name>/*.jpg
      test/test/<class_name>/*.jpg

3) If there is no train folder, a labeled folder such as test/test/<class_name>/
   is split into training and validation subsets. This fallback is convenient
   for datasets that only have one labeled directory, but its validation score
   is NOT an independent test score.
"""
from pathlib import Path
import argparse
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_imagefolder_dir(candidates):
    for path in candidates:
        if path.is_dir():
            class_dirs = [p for p in path.iterdir() if p.is_dir()]
            if class_dirs:
                has_images = any(
                    f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                    for folder in class_dirs for f in folder.iterdir()
                )
                if has_images:
                    return path
    return None


def build_model(num_classes, pretrained):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Facial Expression DataSet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_dir)
    if not data_root.is_absolute():
        data_root = project_dir / data_root
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    train_dir = find_imagefolder_dir([
        data_root / "train" / "train",
        data_root / "train",
    ])
    test_dir = find_imagefolder_dir([
        data_root / "test" / "test",
        data_root / "test",
    ])

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    if train_dir:
        train_dataset = datasets.ImageFolder(str(train_dir), transform=transform)
        if len(train_dataset.classes) < 2:
            raise ValueError("At least two expression class folders are required.")
        if test_dir:
            test_dataset = datasets.ImageFolder(str(test_dir), transform=transform)
            if train_dataset.classes != test_dataset.classes:
                raise ValueError(
                    "Train/test class folders do not match.\n"
                    f"Train: {train_dataset.classes}\nTest: {test_dataset.classes}"
                )
        else:
            # Hold out a validation subset from the training folder.
            val_size = max(1, int(0.2 * len(train_dataset)))
            train_size = len(train_dataset) - val_size
            if train_size < 1:
                raise ValueError("Not enough images to create train/validation splits.")
            train_dataset, test_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(args.seed)
            )
            print("WARNING: No separate test folder found; reporting validation accuracy.")
    else:
        # Fallback for the source notebook's test/test-only dataset.
        if not test_dir:
            raise FileNotFoundError(
                "Could not find labeled image folders. Expected train/<class>/ "
                "and/or test/<class>/ under: " + str(data_root)
            )
        full_dataset = datasets.ImageFolder(str(test_dir), transform=transform)
        if len(full_dataset.classes) < 2:
            raise ValueError("At least two expression class folders are required.")
        val_size = max(1, int(0.2 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        if train_size < 1:
            raise ValueError("Not enough images to create an 80/20 split.")
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(
            "WARNING: Training and validation are split from the same labeled "
            "folder. This is not an independent test evaluation."
        )

    # Get class names from the original ImageFolder dataset.
    if train_dir:
        class_names = datasets.ImageFolder(str(train_dir)).classes
    else:
        class_names = full_dataset.classes

    if len(class_names) < 2:
        raise ValueError("At least two classes are required.")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_dataset)} | Validation/test images: {len(test_dataset)}")

    model = build_model(len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={total_loss / max(total, 1):.4f} | "
            f"accuracy={100 * correct / max(total, 1):.2f}%"
        )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    score = 100 * correct / max(total, 1)
    print(f"Validation/test accuracy: {score:.2f}%")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "architecture": "resnet18",
        "model_state_dict": model.cpu().state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
    }
    checkpoint_path = output_dir / "resnet18_expression_model.pth"
    torch.save(checkpoint, checkpoint_path)

    with (output_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved model checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
