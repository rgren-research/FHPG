import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# from datasets import glue_dataloader
# from datasets import squad_test_dataloader
from evaluate.cifar10 import eval_cifar_acc
# from evaluate.squad import eval_squad_acc
from logger import create_logger
from dataset import build_dataset

# dataset_val, _ = build_dataset(is_train=False, args=args)


@torch.no_grad()
def test_accuracy(model, head_mask, neuron_mask, task_name):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
    ])
    # Loading the CIFAR dataset
    if task_name.lower() == 'cifar':
        dataset = datasets.CIFAR10(root='/data', train=False, download=False, transform=transform)
    elif task_name.lower() == 'cifar100':
        dataset = datasets.CIFAR100(root='/data', train=False, download=False, transform=transform)
    else:
        raise ValueError("Unsupported task_name. Expected 'CIFAR10' or 'CIFAR100'.")
    # Create a data loader
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    # Set the model to evaluation mode.
    model.eval()
    print("Start calculating accuracy (Acc)")
    acc = eval_cifar_acc(
        model,
        head_mask,
        neuron_mask,
        test_loader,
        task_name,
    )
    print("Acc calculation complete!")
    return acc