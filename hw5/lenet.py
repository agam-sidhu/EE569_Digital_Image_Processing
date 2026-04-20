"""
Name: Agam Sidhu
USC ID: 3027948957
USC Email: agamsidh@usc.edu
Submission Date: April 19th, 2026
Problem 1(b): Compare classification performance on different datasets
Problem 1(c): Evaluation and Ablation Study
Problem 1(d):Classification with noisy data 
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import (
    LabelOverrideDataset,
    apply_symmetric_label_noise,
    compute_roc_auc,
    count_parameters,
    ensure_dir,
    get_top_confused_pairs,
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_error_bar_curve,
    save_confused_examples,
    save_csv,
    save_json,
    select_device,
    set_seed,
)

#Directory setup of project
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATA_ROOT = OUTPUT_ROOT / "datasets"
#Classes for FashionMNIST
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
#Classes for CIFAR
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
# Information of about each dataset
DATASET_INFO = {
    "mnist": {
        "dataset_class": datasets.MNIST,
        "channels": 1,
        "image_size": 28,
        "num_classes": 10,
        "classes": [str(index) for index in range(10)],
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashion": {
        "dataset_class": datasets.FashionMNIST,
        "channels": 1,
        "image_size": 28,
        "num_classes": 10,
        "classes": FASHION_CLASSES,
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "cifar": {
        "dataset_class": datasets.CIFAR10,
        "channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "classes": CIFAR_CLASSES,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
}
#Settings of hyperparms
HYPERPARAMETER_SETTINGS = {
    "mnist": [
        {
            "name": "setting_1",
            "init": "kaiming_uniform",
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        {
            "name": "setting_2",
            "init": "xavier_uniform",
            "optimizer": "adam",
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "lr_decay_gamma": 0.5,
            "lr_decay_fractions": (0.7,),
        },
        {
            "name": "setting_3",
            "init": "trunc_normal",
            "optimizer": "sgd",
            "lr": 1e-2,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "lr_decay_gamma": 0.2,
            "lr_decay_fractions": (0.6, 0.8),
        },
    ],
    "fashion": [
        {
            "name": "setting_1",
            "init": "kaiming_uniform",
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        {
            "name": "setting_2",
            "init": "xavier_uniform",
            "optimizer": "adam",
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "lr_decay_gamma": 0.5,
            "lr_decay_fractions": (0.7,),
        },
        {
            "name": "setting_3",
            "init": "trunc_normal",
            "optimizer": "sgd",
            "lr": 2e-2,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
            "lr_decay_gamma": 0.2,
            "lr_decay_fractions": (0.6, 0.8),
        },
    ],
    "cifar": [
        {
            "name": "setting_1",
            "init": "kaiming_uniform",
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "name": "setting_2",
            "init": "xavier_uniform",
            "optimizer": "adam",
            "lr": 5e-4,
            "weight_decay": 5e-4,
            "lr_decay_gamma": 0.5,
            "lr_decay_fractions": (0.7,),
        },
        {
            "name": "setting_3",
            "init": "trunc_normal",
            "optimizer": "sgd",
            "lr": 3e-2,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
            "lr_decay_gamma": 0.2,
            "lr_decay_fractions": (0.6, 0.8),
        },
    ],
}
#Eposchs set for each dataset
DEFAULT_EPOCHS = {
    "mnist": 10,
    "fashion": 25,
    "cifar": 20,
}


# Model Definition
class LeNet5(nn.Module):
    def __init__(self, input_channels, image_size, num_classes=10):
        super().__init__()
         # First convol layer (input -> 6 feature maps (5x5 kernel) )
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=0)
        # 2x2 max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convol layer (6 feature maps -> 16 feature maps (5x5 kernel) )
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        #compute feature dimension
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, image_size, image_size)
            sample = self.pool(torch.relu(self.conv1(sample)))
            sample = self.pool(torch.relu(self.conv2(sample)))
            feature_dim = sample.view(1, -1).shape[1]
        #fully connected layers 
        self.fc1 = nn.Linear(feature_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    #Forward Pass Function
    def forward(self, inputs):
        #Apply first convol + pool + ReLU
        outputs = self.pool(torch.relu(self.conv1(inputs)))
        #Apply second convol + pool + ReLU
        outputs = self.pool(torch.relu(self.conv2(outputs)))
        #Flattens into vector
        outputs = torch.flatten(outputs, start_dim=1)
        #Pass through 1st layer of fc
        outputs = torch.relu(self.fc1(outputs))
        #Pass through 2nd layer of fc
        outputs = torch.relu(self.fc2(outputs))
        # FInal Output layer
        return self.fc3(outputs)


# Data Loading Functions
def get_hyperparms(dataset_name, setting_name):
    #Gets the hyperparams for a dataset + setting
    settings = HYPERPARAMETER_SETTINGS[dataset_name]
    #Returns all settings
    if setting_name == "all":
        return settings
    #Returns setting by index
    if setting_name.isdigit():
        s_idx = int(setting_name) - 1
        if 0 <= s_idx < len(settings):
            return [settings[s_idx]]
    #Picks setting by name
    for setting in settings:
        if setting["name"] == setting_name:
            return [setting]
    #In case we get incorrect setting input
    raise ValueError(f"Unknown setting '{setting_name}' for dataset '{dataset_name}'.")

def build_transforms(dataset_name, is_train):
    #Get dataset info
    dataset_info = DATASET_INFO[dataset_name]
    trans_list = []

    # Adds training augementation for Fashion and CIFAR
    if is_train and dataset_name == "fashion":
        trans_list.extend(
            [
                transforms.RandomCrop(dataset_info["image_size"], padding=2),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif is_train and dataset_name == "cifar":
        trans_list.extend(
            [
                transforms.RandomCrop(dataset_info["image_size"], padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    #Convert to tensor + normalize
    trans_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(dataset_info["mean"], dataset_info["std"]),
        ]
    )
    return transforms.Compose(trans_list)


def build_dataloaders(
    dataset_name,
    batch_size,
    num_workers,
    noise_epsilon=None,
    noise_seed=0,
):
    #Get dataset info + class
    dataset_info = DATASET_INFO[dataset_name]
    dataset_class = dataset_info["dataset_class"]

    # Build training and testing transforms
    train_transform = build_transforms(dataset_name, is_train=True)
    test_transform = build_transforms(dataset_name, is_train=False)

    #Loads training + testing dataset
    train_dataset = dataset_class(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = dataset_class(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform,
    )

    noise_info = None
    if noise_epsilon is not None:
        # Apply symmetric label noise to training set (labels)
        og_targets = np.asarray(train_dataset.targets, dtype=np.int64)
        noisy_targets, clean_targets, changed_mask = apply_symmetric_label_noise(
            og_targets,
            noise_epsilon,
            dataset_info["num_classes"],
            noise_seed,
        )
        train_dataset = LabelOverrideDataset(train_dataset, noisy_targets)
        noise_info = {
            "true_labels": clean_targets.tolist(),
            "noisy_labels": noisy_targets.tolist(),
            "changed_fraction": float(changed_mask.mean()),
        }

    #Builds the dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, dataset_info, noise_info


# Optimization Functions
def build_optimizer(model, setting):
    #Builds the optimizer bassed on the selected setting
    if setting["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=setting["lr"],
            momentum=setting.get("momentum", 0.9),
            weight_decay=setting.get("weight_decay", 0.0),
            nesterov=setting.get("nesterov", False),
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=setting["lr"],
        weight_decay=setting.get("weight_decay", 0.0),
    )

#Intializes the model
def initialize(model, setting):
    # Initialize convolution and linear layers based on the selected setting.
    init_name = setting.get("init", "kaiming_uniform")
    #Intializes the model (convolution and linear layers)
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if init_name == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif init_name == "trunc_normal":
                nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.05)
            else:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

#LR Scheduler 
def build_scheduler(optimizer, setting, epochs):
    # Gets lr decay settings
    gamma = setting.get("lr_decay_gamma")
    fractions = setting.get("lr_decay_fractions")
    if gamma is None or not fractions or epochs < 4:
        return None
    #Convert fractions to milestones
    milestone = sorted({max(1, int(epochs * fraction)) for fraction in fractions})
    #returns the scheduler
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)


# Training Functions
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    #loops over our training batches
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        #Our forward pass + loss
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        #backprop
        loss.backward()
        optimizer.step()

        #update stats
        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        
    avg_loss = total_loss / total_examples
    accuracy = 100.0 * total_correct / total_examples
 
    return avg_loss, accuracy


# Evaluation Functions
def evaluate(model, data_loader, criterion, device, collect_outputs=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0


    testLabels = []
    testPreds = []
    testProbs = []
    testimages = []

    #loops over the evaluation batches
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            #calculate loss
            if criterion is not None:
                total_loss += criterion(outputs, labels).item() * labels.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            #update stats
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            #saves outputs (confusion matrix + ROC)
            if collect_outputs:
                testLabels.append(labels.cpu())
                testPreds.append(preds.cpu())
                testProbs.append(probs.cpu())
                testimages.append(inputs.cpu())
    #returns loss + accuracy
    results = {
        "loss": total_loss / total_examples if criterion is not None else None,
        "accuracy": 100.0 * total_correct / total_examples,
    }
    #saves outputs
    if collect_outputs:
        results["labels"] = torch.cat(testLabels).numpy()
        results["predictions"] = torch.cat(testPreds).numpy()
        results["probabilities"] = torch.cat(testProbs).numpy()
        results["images"] = torch.cat(testimages)

    return results


# Experiment Functions
def run_one_training(
    dataset_name,
    setting,
    epochs,
    batch_size,
    num_workers,
    device,
    seed,
    run_dir,
    noise_epsilon=None,
):
    #sets rnadom seed
    set_seed(seed)
    #setts up output folder
    ensure_dir(run_dir)

    # Build dataloaders and model for one iteration
    train_loader, test_loader, dataset_info, noise_info = build_dataloaders(
        dataset_name,
        batch_size,
        num_workers,
        noise_epsilon=noise_epsilon,
        noise_seed=seed,
    )

    #builds model + optimizer (setup)
    model = LeNet5(
        input_channels=dataset_info["channels"],
        image_size=dataset_info["image_size"],
        num_classes=dataset_info["num_classes"],
    ).to(device)
    #intializes model
    initialize(model, setting)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, setting)
    scheduler = build_scheduler(optimizer, setting, epochs)

    train_acc = []
    test_acc = []
    best_acc = 0.0
    best_state = None

    #trains based on # of epochs
    for epoch_index in range(epochs):
        _, train_accuracy =  train_epoch(model, train_loader, criterion, optimizer, device)
        eval = evaluate(model, test_loader, criterion, device, collect_outputs=False)

        train_acc.append(train_accuracy)
        test_acc.append(eval["accuracy"])

        #saves best model (by test accuracy)
        if eval["accuracy"] >= best_acc:
            best_acc = eval["accuracy"]
            best_state = deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()
        #Shows training progress for current run
        progress_percent = 100.0 * (epoch_index + 1) / epochs
        print(
            f"Epoch {epoch_index + 1}/{epochs} "
            f"({progress_percent:.1f}%) "
            f"train_acc={train_accuracy:.2f}% "
            f"test_acc={eval['accuracy']:.2f}%",
            flush=True,
        )
    #stores it in the folder
    model_path = run_dir / "best_model.pt"
    torch.save(best_state, model_path)

    #saves metrics
    run_metrics = {
        "seed": seed,
        "setting": setting["name"],
        "initialization": setting.get("init"),
        "optimizer": setting["optimizer"],
        "lr": setting["lr"],
        "weight_decay": setting.get("weight_decay", 0.0),
        "lr_decay_gamma": setting.get("lr_decay_gamma"),
        "lr_decay_fractions": list(setting.get("lr_decay_fractions", [])),
        "epochs": epochs,
        "best_accuracy": best_acc,
        "final_accuracy": test_acc[-1],
        "parameter_count": count_parameters(model),
    }
    #stores metrics
    save_json(run_metrics, run_dir / "metrics.json")

    return {
        "metrics": run_metrics,
        "train_accuracies": train_acc,
        "test_accuracies": test_acc,
        "model_path": model_path,
        "noise_info": noise_info,
    }

# Analysis Functions
def analyze_best(dataset_name, best_entry, batch_size, num_workers, device):
    # Gets best setting & run information
    setting = best_entry["setting"]
    best_run = best_entry["best_run_result"]
    run_results = best_entry["run_results"]
    analysis_dir = ensure_dir(OUTPUT_ROOT / dataset_name / "best_setting_analysis")

    #builds dataloder
    _, test_loader, dataset_info, _ = build_dataloaders(dataset_name, batch_size, num_workers)

    #loads best model
    model = LeNet5(
        input_channels=dataset_info["channels"],
        image_size=dataset_info["image_size"],
        num_classes=dataset_info["num_classes"],
    ).to(device)
    model.load_state_dict(torch.load(best_run["model_path"], map_location=device))
    criterion = nn.CrossEntropyLoss()
    evaluation = evaluate(model, test_loader, criterion, device, collect_outputs=True)

    # Calculates normalized confusion matrix
    confusion = plot_confusion_matrix(
        evaluation["labels"],
        evaluation["predictions"],
        dataset_info["classes"],
        analysis_dir / "confusion_matrix.png",
        f"{dataset_name.upper()} normalized confusion matrix",
        normalize="true",
    )
    #Saves 1 example image for each pair
    confused_pairs = get_top_confused_pairs(confusion, dataset_info["classes"], top_k=3)
    saved_examples = save_confused_examples(
        evaluation["images"],
        evaluation["labels"],
        evaluation["predictions"],
        confused_pairs,
        dataset_info["classes"],
        dataset_info["mean"],
        dataset_info["std"],
        analysis_dir / "confused_examples",
    )
    # Summarizes analysis
    analysis_summary = {
        "chosen_setting": setting["name"],
        "chosen_by": "highest mean accuracy across the 5 runs",
        "top_confused_pairs": confused_pairs,
        "saved_example_images": saved_examples,
    }
    #Adds ROC AUC for cifar
    if dataset_name == "cifar":
        roc_summary = compute_roc_auc(
            evaluation["labels"],
            evaluation["probabilities"],
            dataset_info["classes"],
            analysis_dir / "roc_curves.png",
        )
        #Calculates average weighted AUC across the 5 runs
        auc_values = []
        for run_result in run_results:
            run_model = LeNet5(
                input_channels=dataset_info["channels"],
                image_size=dataset_info["image_size"],
                num_classes=dataset_info["num_classes"],
            ).to(device)
            run_model.load_state_dict(torch.load(run_result["model_path"], map_location=device))
            run_eval = evaluate(run_model, test_loader, criterion, device, collect_outputs=True)
            weighted_auc = roc_auc_score(
                run_eval["labels"],
                run_eval["probabilities"],
                multi_class="ovr",
                average="weighted",
            )
            auc_values.append(float(weighted_auc))
        analysis_summary["roc_auc"] = {
            "curve_basis": "representative best run within the chosen setting",
            "representative_run_weighted_auc": roc_summary["weighted_auc"],
            "representative_run_per_class_auc": roc_summary["per_class_auc"],
            "weighted_auc_mean_5_runs": float(np.mean(auc_values)),
            "weighted_auc_std_5_runs": float(np.std(auc_values)),
            "weighted_auc_per_run": auc_values,
        }

    save_json(analysis_summary, analysis_dir / "analysis_summary.json")

#Runs the lenet model
def run_lenet(dataset_name, args):
    # Run all selected settings and summarize the results
    device = select_device(args.device)
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[dataset_name]
    #Gets hyper param settings
    settings = get_hyperparms(dataset_name, args.setting)
    dataset_output_dir = ensure_dir(OUTPUT_ROOT / dataset_name)

    sum_rows = []
    best_entry = None

    #Runs the model based on settings
    for setting in settings:
        print(f"Running {dataset_name} with {setting['name']} on {device}...")
        setting_dir = ensure_dir(dataset_output_dir / setting["name"])
        run_results = []
        #Runs the setting multiple times
        for run_index in range(args.runs):
            seed = 2026 + run_index
            run_dir = ensure_dir(setting_dir / f"run_{run_index + 1}")
            result = run_one_training(
                dataset_name=dataset_name,
                setting=setting,
                epochs=epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                seed=seed,
                run_dir=run_dir,
                noise_epsilon=None,
            )
            run_results.append(result)

            if run_index == 0:
                # Save one accuracy curve per setting
                plot_accuracy_curves(
                    result["train_accuracies"],
                    result["test_accuracies"],
                    f"{dataset_name.upper()} {setting['name']} accuracy curve",
                    setting_dir / "accuracy_curve.png",
                )
        #Summarizes the results
        run_accuracies = [result["metrics"]["best_accuracy"] for result in run_results]
        best_run_result = max(run_results, key=lambda item: item["metrics"]["best_accuracy"])
        summary_row = {
            "dataset": dataset_name,
            "setting": setting["name"],
            "initialization": setting.get("init"),
            "optimizer": setting["optimizer"],
            "lr": setting["lr"],
            "weight_decay": setting.get("weight_decay", 0.0),
            "lr_decay_gamma": setting.get("lr_decay_gamma"),
            "best_accuracy": float(np.max(run_accuracies)),
            "mean_accuracy": float(np.mean(run_accuracies)),
            "std_accuracy": float(np.std(run_accuracies)),
        }
        sum_rows.append(summary_row)
        save_json(summary_row, setting_dir / "summary.json")
        #Tracks best setting
        if best_entry is None or summary_row["mean_accuracy"] > best_entry["summary"]["mean_accuracy"]:
            best_entry = {
                "summary": summary_row,
                "setting": setting,
                "best_run_result": best_run_result,
                "run_results": run_results,
            }
    #Saves overall summary table
    save_csv(sum_rows, dataset_output_dir / "setting_summary.csv")
    save_json(sum_rows, dataset_output_dir / "setting_summary.json")
    #Analyzes the best setting
    if best_entry is not None:
        analyze_best(
            dataset_name,
            best_entry,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

#Noise functions
def save_noise_matrix(dataset_name, batch_size, num_workers):
    # Builds noisy label at episilon = 50%
    _, _, dataset_info, noise_info = build_dataloaders(
        dataset_name,
        batch_size,
        num_workers,
        noise_epsilon=50,
        noise_seed=2026,
    )
    noise_dir = ensure_dir(OUTPUT_ROOT / "noise" / dataset_name)
    #saves normalized confusion matrix
    plot_confusion_matrix(
        noise_info["true_labels"],
        noise_info["noisy_labels"],
        dataset_info["classes"],
        noise_dir / "epsilon_50_true_vs_noisy_confusion.png",
        f"{dataset_name.upper()} true vs noisy labels (epsilon=50%)",
        normalize="true",
    )
    #save summary
    save_json(
        {
            "epsilon": 50,
            "changed_fraction": noise_info["changed_fraction"],
        },
        noise_dir / "epsilon_50_noise_summary.json",
    )

# Runs the noise approach
def run_noise(args):
    # Get dataset name + training
    dataset_name = args.noise_dataset
    device = select_device(args.device)
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[dataset_name]
    noise_levels = [0, 20, 40, 60, 80]
    noise_dir = ensure_dir(OUTPUT_ROOT / "noise" / dataset_name)
    #selects the setting of hyperparams
    if args.setting == "all":
        setting = HYPERPARAMETER_SETTINGS[dataset_name][0]
    else:
        setting = get_hyperparms(dataset_name, args.setting)[0]

    rows = []
    mean_acc = []
    std_acc = []
    #Runs all noise levels
    for epsilon in noise_levels:
        print(f"Running noise experiment for {dataset_name}, epsilon={epsilon}%...")
        epsilon_dir = ensure_dir(noise_dir / f"epsilon_{epsilon}")
        run_accuracies = []
        #runs same noise multiple times
        for run_index in range(args.runs):
            seed = 3027 + run_index
            run_dir = ensure_dir(epsilon_dir / f"run_{run_index + 1}")
            result = run_one_training(
                dataset_name=dataset_name,
                setting=setting,
                epochs=epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                seed=seed,
                run_dir=run_dir,
                noise_epsilon=epsilon,
            )
            run_accuracies.append(result["metrics"]["best_accuracy"])
            #Saves one accuracy curve per run
            if run_index == 0:
                plot_accuracy_curves(
                    result["train_accuracies"],
                    result["test_accuracies"],
                    f"{dataset_name.upper()} epsilon={epsilon}% accuracy curve",
                    epsilon_dir / "accuracy_curve.png",
                )

        summary_row = {
            "dataset": dataset_name,
            "setting": setting["name"],
            "initialization": setting.get("init"),
            "epsilon": epsilon,
            "best_accuracy": float(np.max(run_accuracies)),
            "mean_accuracy": float(np.mean(run_accuracies)),
            "std_accuracy": float(np.std(run_accuracies)),
        }
        rows.append(summary_row)
        mean_acc.append(summary_row["mean_accuracy"])
        std_acc.append(summary_row["std_accuracy"])
        save_json(summary_row, epsilon_dir / "summary.json")
    #saves accuracy vs epsilon
    plot_error_bar_curve(
        noise_levels,
        mean_acc,
        std_acc,
        f"{dataset_name.upper()} accuracy vs symmetric label noise",
        "Noise level epsilon (%)",
        "Test accuracy (%)",
        noise_dir / "accuracy_vs_epsilon.png",
    )
    #saves noise summary
    save_csv(rows, noise_dir / "noise_summary.csv")
    save_json(rows, noise_dir / "noise_summary.json")
    #saves epsilon (50%)
    save_noise_matrix(dataset_name, args.batch_size, args.num_workers)


def run_lenet_experiment(dataset_name, args):
    return run_lenet(dataset_name, args)


def run_noise_experiment(args):
    return run_noise(args)
