import sys
# tips:Remove specific loading paths


import torch
from pruner.fisher import apply_neuron_mask
from datasets import load_metric
from transformers import ViTConfig
from transformers import DeiTConfig
import torch.nn as nn




@torch.no_grad()
def eval_cifar_acc(model, head_mask, neuron_mask, dataloader, task_name):
    metric = load_metric("accuracy")
    model.eval()
    all_preds = []
    all_labels = []

    handles = apply_neuron_mask(model, neuron_mask)

    config = DeiTConfig.from_pretrained(
        '/retraining-free-pruning-main/deit-base')
    hidden_size = config.hidden_size
    # num_classes
    num_classes = 10  # Replace according to the actual situation.
    classification_head = nn.Linear(hidden_size, num_classes)
    classification_head = classification_head.cuda()  # Ensure the classification head is on the GPU

    for batch in dataloader:
        inputs, _ = batch  # Assuming the data loader returns (input, target), we only need the input.
        inputs = inputs.to("cuda", non_blocking=True)
        #
        outputs = model(inputs, head_mask=head_mask)

        # Assume outputs.logits is the model's predicted logits, and labels is the actual labels.
        labels = batch[1].to("cuda")
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        # Then use the classification header to obtain logits.
        logits = classification_head(pooled_output)
        # Step 1: Obtain the predicted category from logits
        _, preds = torch.max(logits, dim=1)
        # Before expanding to all_labels, move labels to the CPU and convert them to NumPy arrays.
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())  # The labels were copied from the GPU to CPU memory.

        # Step 2: Calculate the number of correctly predicted samples
        # correct_preds = torch.sum(preds == labels)
        # Step 3: Calculate the accuracy rate
        # accuracy = correct_preds.float() / labels.size(0)
        # Printing accuracy
        # print(f"Accuracy: {accuracy.item() * 100:.2f}%")
    for handle in handles:
        handle.remove()
    head_mask.requires_grad_(False)  # It does not participate in gradient calculation, thus avoiding unnecessary gradient computation.
    neuron_mask.requires_grad_(False)

    eval_results = metric.compute(predictions=all_preds, references=all_labels)
    accuracy = eval_results["accuracy"]
    print("Complete based on " + task_name + " Evaluation of the model after task pruning.")
    # eval_results = metric.compute()
    # target_metric = target_dev_metric(task_name)
    # accuracy = eval_results[target_metric]
    return accuracy


