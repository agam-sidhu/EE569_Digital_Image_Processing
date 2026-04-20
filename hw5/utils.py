"""
Name: Agam Sidhu
USC ID: 3027948957
USC Email: agamsidh@usc.edu
Submission Date: April 19th, 2026
Helper Functions: Problems 1(b), 1(c), 1(d), and 2(b)
"""

import csv
import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset

# Basic Helper Functions

def ensure_dir(path):
    #Creates directory if it doesnt exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_filename(text):
    #Cleans filename so it can be saved safely
    text = re.sub(r'[<>:"/\\|?*\s]+', "_", str(text).strip())
    return text.strip("._") or "item"


def set_seed(seed):
    #Sets all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_name):
    #Selects cpu or cuda
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Saving Functions

def save_json(data, path):
    #Saves dictionary/list data as json
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(rows, path):
    #Saves rows as csv
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Plotting Functions

def plot_accuracy_curves(train_accuracies, test_accuracies, title, path):
    #Plots train and test accuracy curves
    epochs = np.arange(1, len(train_accuracies) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, marker="o", label="Train accuracy")
    plt.plot(epochs, test_accuracies, marker="s", label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

#Plots std error bars + mean accuracy
def plot_error_bar_curve(x_values, means, stds, title, xlabel, ylabel, path):
    plt.figure(figsize=(8, 5))
    plt.errorbar(x_values, means, yerr=stds, fmt="-o", capsize=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

#Builds conufusion matrix
def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    class_names,
    path,
    title,
    normalize="true",
):
    #Computes and plots confusion matrix
    class_indices = list(range(len(class_names)))
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_indices,
        normalize=normalize,
    )

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(ticks=class_indices, labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=class_indices, labels=class_names)
    plt.colorbar()

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            plt.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return matrix


# Analysis Functions

def get_top_confused_pairs(confusion, class_names, top_k=3):
    #Finds the top confused class pairs
    
    confusion = np.array(confusion, copy=True)
    np.fill_diagonal(confusion, 0.0)
    flat_indices = np.argsort(confusion.ravel())[::-1]

    confused_pairs = []
    for flat_index in flat_indices:
        row, col = np.unravel_index(flat_index, confusion.shape)
        if confusion[row, col] <= 0:
            break
        confused_pairs.append(
            {
                "true_index": int(row),
                "pred_index": int(col),
                "true_class": class_names[row],
                "pred_class": class_names[col],
                "score": float(confusion[row, col]),
            }
        )
        if len(confused_pairs) == top_k:
            break
    return confused_pairs

#Converts normalized tensor back to image range
def denormalize_image(image_tensor, mean, std):
    #Converts normalized tensor back to image range
    image_tensor = image_tensor.detach().cpu().clone()
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype).view(-1, 1, 1)
    image_tensor = image_tensor * std_tensor + mean_tensor
    image_tensor = image_tensor.clamp(0.0, 1.0)

    if image_tensor.shape[0] == 1:
        return image_tensor.squeeze(0).numpy(), "gray"
    return image_tensor.permute(1, 2, 0).numpy(), None

#Saves one example image for each confused pair
def save_confused_examples(
    images,
    true_labels,
    predicted_labels,
    confused_pairs,
    class_names,
    mean,
    std,
    output_dir,
):
    output_dir = ensure_dir(output_dir)
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    saved = []
    for pair in confused_pairs:
        mask = np.where(
            (true_labels == pair["true_index"]) & (predicted_labels == pair["pred_index"])
        )[0]
        if len(mask) == 0:
            continue

        sample_index = int(mask[0])
        image, cmap = denormalize_image(images[sample_index], mean, std)
        safe_name = clean_filename(f"{pair['true_class']}_as_{pair['pred_class']}")
        path = output_dir / f"{safe_name}.png"

        plt.figure(figsize=(3, 3))
        if cmap:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.title(f"True: {pair['true_class']}\nPred: {pair['pred_class']}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        saved.append({"path": str(path), **pair})
    return saved

#Computes one v rest ROC curves and weighted AUC
def compute_roc_auc(true_labels, probabilities, class_names, plot_path):
    true_labels = np.asarray(true_labels)
    probabilities = np.asarray(probabilities)
    binary_labels = label_binarize(true_labels, classes=np.arange(len(class_names)))

    plt.figure(figsize=(9, 7))
    per_class_auc = {}
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_index], probabilities[:, class_index])
        auc_value = roc_auc_score(binary_labels[:, class_index], probabilities[:, class_index])
        per_class_auc[class_name] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CIFAR-10 One-vs-Rest ROC Curves")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    weighted_auc = roc_auc_score(
        true_labels,
        probabilities,
        multi_class="ovr",
        average="weighted",
    )
    return {
        "weighted_auc": float(weighted_auc),
        "per_class_auc": per_class_auc,
    }


# Label Noise Functions
def apply_symmetric_label_noise(labels, epsilon_percent, num_classes, seed):
    #Applies symmetric label noise to labels
    labels = np.asarray(labels, dtype=np.int64)
    noisy_labels = labels.copy()
    rng = np.random.default_rng(seed)

    if epsilon_percent <= 0:
        changed_mask = np.zeros(len(labels), dtype=bool)
        return noisy_labels, labels.copy(), changed_mask

    changed_mask = rng.random(len(labels)) < (epsilon_percent / 100.0)
    changed_indices = np.where(changed_mask)[0]

    for index in changed_indices:
        choices = [label for label in range(num_classes) if label != int(labels[index])]
        noisy_labels[index] = int(rng.choice(choices))

    return noisy_labels, labels.copy(), changed_mask


# Dataset Helper Class

class LabelOverrideDataset(Dataset):
    def __init__(self, base_dataset, labels):
        self.base_dataset = base_dataset
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes = getattr(base_dataset, "classes", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        return image, int(self.labels[index])


# Model Helper Function
def count_parameters(model):
    #Counts the total number of parameters
    return int(sum(parameter.numel() for parameter in model.parameters()))
