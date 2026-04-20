EE569 Homework #5 Coding Files

This submission keeps only the required source files. Outputs are not included per homework instructions and can be regenerated using the provided code.

Files

main.py
- Command-line entry point for all homework experiments.

lenet.py
- LeNet-5 code for Problem 1.
- Includes MNIST, Fashion-MNIST, and CIFAR-10 training.
- Includes the required noisy-label experiment for Fashion-MNIST.

vit.py
- Wrapper code for Problem 2(b).
- This part uses a referenced implementation [9]. The code automatically clones the repository during execution. The reference code is not included in this submission.

utils.py
- Shared helper functions for plotting, JSON/CSV saving, confusion matrices, ROC/AUC, label noise, and device selection.

requirements.txt
- Python package requirements.

How to install

pip install -r requirements.txt

How to run

Run these commands from inside the hw5 folder.

python main.py --experiment mnist
python main.py --experiment fashion
python main.py --experiment cifar
python main.py --experiment noise --noise-dataset fashion
python main.py --experiment vit

How to run on Google Colab

After uploading and extracting the hw5 folder in Colab:

cd /content/hw5
python -m pip install -r requirements.txt

python main.py --experiment mnist --device cuda
python main.py --experiment fashion --device cuda
python main.py --experiment cifar --device cuda
python main.py --experiment noise --noise-dataset fashion --device cuda
python main.py --experiment vit --device cuda

If CUDA is not available in the Colab runtime, the code falls back to CPU.

Problem 2(b) note

This part uses a referenced implementation [9]. The code automatically clones the repository during execution. The reference code is not included in this submission.

Reproducibility note

Outputs are not included per homework instructions and can be regenerated using the provided code.
