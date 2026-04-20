"""
Name: Agam Sidhu
USC ID: 3027948957
USC Email: agamsidh@usc.edu
Submission Date: April 19th, 2026
Problem 2(b): Compare classification performance over the MNIST dataset
"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils import (
    count_parameters,
    ensure_dir,
    plot_accuracy_curves,
    save_json,
    select_device,
    set_seed,
)

# Directory setup of project
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vit"
DATA_ROOT = PROJECT_ROOT / "outputs" / "datasets"

# Reference repo setup
REFERENCE_REPO_URL = "https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT.git"
REFERENCE_REPO_NAMES = [
    "PyTorch-Scratch-Vision-Transformer-ViT-main",
    "PyTorch-Scratch-Vision-Transformer-ViT",
]


# Reference repo helper functions
def find_reference_repo():
    # Reuse an existing local reference repo when available
    for repo_name in REFERENCE_REPO_NAMES:
        repo_dir = PROJECT_ROOT / repo_name
        if (repo_dir / "data_loader.py").exists() and (repo_dir / "model.py").exists():
            return repo_dir
    return None


def ensure_reference_repo():
    # Clone the allowed reference repo only if it is missing
    repo_dir = find_reference_repo()
    if repo_dir is not None:
        return repo_dir

    repo_dir = PROJECT_ROOT / REFERENCE_REPO_NAMES[0]
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REFERENCE_REPO_URL, str(repo_dir)],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git is required to clone the ViT reference repo.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to clone the ViT reference repo.") from error

    return repo_dir


def load_reference_module(module_name, module_path):
    # Load a python module directly from the reference repo
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference_components():
    # Load the external data loader and model code
    repo_dir = ensure_reference_repo()
    data_loader_module = load_reference_module(
        "vit_reference_data_loader",
        repo_dir / "data_loader.py",
    )
    model_module = load_reference_module(
        "vit_reference_model",
        repo_dir / "model.py",
    )
    return repo_dir, data_loader_module, model_module


def build_reference_args(batch_size, num_workers, epochs, device):
    # Build the reference repo argument set for MNIST
    return SimpleNamespace(
        dataset="mnist",
        epochs=epochs,
        warmup_epochs=10,
        batch_size=batch_size,
        n_classes=10,
        n_workers=num_workers,
        lr=5e-4,
        output_path=str(OUTPUT_ROOT / "reference_outputs"),
        image_size=28,
        patch_size=4,
        n_channels=1,
        data_path=str(DATA_ROOT),
        use_torch_transformer_layers=False,
        embed_dim=64,
        n_attention_heads=4,
        forward_mul=2,
        n_layers=6,
        dropout=0.1,
        model_path=str(OUTPUT_ROOT / "reference_model"),
        load_model=False,
        n_patches=(28 // 4) ** 2,
        is_cuda=(device.type == "cuda"),
    )


def build_reference_model(model_module, ref_args, device):
    # Build the ViT model from the allowed reference repo
    if ref_args.use_torch_transformer_layers:
        model = model_module.VisionTransformer_pytorch(
            n_channels=ref_args.n_channels,
            embed_dim=ref_args.embed_dim,
            n_layers=ref_args.n_layers,
            n_attention_heads=ref_args.n_attention_heads,
            forward_mul=ref_args.forward_mul,
            image_size=ref_args.image_size,
            patch_size=ref_args.patch_size,
            n_classes=ref_args.n_classes,
            dropout=ref_args.dropout,
        )
    else:
        model = model_module.VisionTransformer(
            n_channels=ref_args.n_channels,
            embed_dim=ref_args.embed_dim,
            n_layers=ref_args.n_layers,
            n_attention_heads=ref_args.n_attention_heads,
            forward_mul=ref_args.forward_mul,
            image_size=ref_args.image_size,
            patch_size=ref_args.patch_size,
            n_classes=ref_args.n_classes,
            dropout=ref_args.dropout,
        )
    return model.to(device)


# Evaluation functions
def evaluate_model(model, data_loader, criterion, device):
    # Evaluate the model on one loader
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return {
        "accuracy": 100.0 * total_correct / total_examples,
        "loss": total_loss / total_examples,
    }


# Experiment function
def run_vit_experiment(args):
    # Run the MNIST ViT experiment using the allowed reference repo
    device = select_device(args.device)
    epochs = args.epochs if args.epochs is not None else 20
    batch_size = args.batch_size
    num_workers = args.num_workers
    output_dir = ensure_dir(OUTPUT_ROOT)

    set_seed(2026)
    repo_dir, data_loader_module, model_module = load_reference_components()
    ref_args = build_reference_args(batch_size, num_workers, epochs, device)

    train_loader, test_loader = data_loader_module.get_loader(ref_args)
    model = build_reference_model(model_module, ref_args, device)
    criterion = nn.CrossEntropyLoss()

    # Use the same optimizer settings as the allowed reference repo
    optimizer = torch.optim.AdamW(model.parameters(), lr=ref_args.lr, weight_decay=1e-3)

    warmup_epochs = min(ref_args.warmup_epochs, epochs)
    warmup_scheduler = None
    if warmup_epochs > 1:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / warmup_epochs,
            end_factor=1.0,
            total_iters=warmup_epochs - 1,
        )

    cosine_scheduler = None
    if epochs > warmup_epochs:
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-5,
        )

    train_accuracies = []
    test_accuracies = []
    best_acc = 0.0
    best_state = None

    print(f"Running Problem 2(b) ViT on MNIST for {epochs} epochs using {device}...")

    for epoch_index in range(epochs):
        model.train()
        train_correct = 0
        train_examples = 0

        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_examples += labels.size(0)

            if batch_index % 50 == 0 or batch_index == (len(train_loader) - 1):
                batch_accuracy = 100.0 * (predictions == labels).sum().item() / labels.size(0)
                print(
                    f"Ep: {epoch_index + 1}/{epochs}\t"
                    f"It: {batch_index + 1}/{len(train_loader)}\t"
                    f"batch_loss: {loss.item():.4f}\t"
                    f"batch_accuracy: {batch_accuracy:.2f}%"
                )

        train_acc = 100.0 * train_correct / train_examples
        test_result = evaluate_model(model, test_loader, criterion, device)
        test_acc = test_result["accuracy"]

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if test_acc >= best_acc:
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch_index + 1:02d}/{epochs}: "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%"
        )

        if warmup_scheduler is not None and epoch_index < (warmup_epochs - 1):
            warmup_scheduler.step()
        elif cosine_scheduler is not None:
            cosine_scheduler.step()

    torch.save(best_state, output_dir / "best_vit_model.pt")
    plot_accuracy_curves(
        train_accuracies,
        test_accuracies,
        "MNIST ViT accuracy curve",
        output_dir / "accuracy_curve.png",
    )
    save_json(
        {
            "epochs": epochs,
            "final_test_accuracy": test_accuracies[-1],
            "best_test_accuracy": best_acc,
            "parameter_count": count_parameters(model),
            "reference_repo_url": REFERENCE_REPO_URL,
            "reference_repo_path": str(repo_dir),
        },
        output_dir / "vit_summary.json",
    )
