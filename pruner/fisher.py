import torch

import torch.nn.functional as F
import torch.nn as nn
from transformers import ViTModel, ViTConfig, DeiTConfig


# config = ViTConfig.from_pretrained('vit-base-patch16-224')
# model = ViTModel.from_pretrained('vit-base-patch16-224')
# hidden_size = model.config.hidden_size


def collect_mask_grads(model, head_mask, neuron_mask, dataloader):

    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)
    handles = apply_neuron_mask(model, neuron_mask)
    model.eval()
    head_grads = []
    neuron_grads = []

    config = DeiTConfig.from_pretrained(
        '/retraining-free-pruning-main/deit-base')
    hidden_size = config.hidden_size
    num_classes = 10  # Replace according to the actual situation.
    #
    classification_head = nn.Linear(hidden_size, num_classes)
    classification_head = classification_head.cuda()

    for batch in dataloader:
        inputs, _ = batch
        inputs = inputs.to("cuda", non_blocking=True)

        # Note: The input and processing methods for the ViT model differ from those of the standard NLP model.
        outputs = model(inputs, head_mask=head_mask)

        # loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        # loss.backward()

        # Assume outputs.logits is the model's predicted logits, and labels is the actual labels.
        labels = batch[1].to("cuda")  # Make sure the labels are also on the same device.
        # loss = F.cross_entropy(outputs.logits, labels)  # Calculate cross-entropy loss
        # loss.backward()

        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)

        logits = classification_head(pooled_output)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        # Collect the gradient at each step, then clear the gradient to zero to avoid gradient accumulation.
        head_grads.append(head_mask.grad.detach())
        neuron_grads.append(neuron_mask.grad.detach())
        head_mask.grad = None
        neuron_mask.grad = None

    # Cleaning
    for handle in handles:
        handle.remove()
    head_mask.requires_grad_(False)  #
    neuron_mask.requires_grad_(False)

    head_grads = torch.stack(head_grads, dim=0)
    neuron_grads = torch.stack(neuron_grads, dim=0)
    return head_grads, neuron_grads

# Map the output of the linear FFN fully connected layer to the mask.
def apply_neuron_mask(model, neuron_mask):
    num_hidden_layers = neuron_mask.shape[0]
    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask(ffn2, neuron_mask[layer_idx])
        handles.append(handle)
    return handles


def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.output
    return ffn2


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask, inputs[1])
    handle = module.register_forward_pre_hook(hook)
    return handle


def get_layers(model):
    encoder = get_encoder(model)
    layers = encoder.layer
    return layers


def get_encoder(model):
    backbone = get_backbone(model)
    encoder = backbone.encoder
    return encoder


# def get_backbone(model):
#     model_type = model.base_model_prefix
#     backbone = getattr(model, model_type)
#     return backbone
def get_backbone(model):
    return model


def get_mha(model, index):
    layer = get_layers(model)[index]
    # Try accessing the attention property directly
    mha = layer.attention
    return mha


def register_head_mask(module, mask):
    def hook(module, inputs):

        q, k, v = inputs[:3]
        # We need to apply the mask to the num_heads dimension.
        q = q * mask.unsqueeze(0).unsqueeze(-1)
        k = k * mask.unsqueeze(0).unsqueeze(-1)
        v = v * mask.unsqueeze(0).unsqueeze(-1)
        return (q, k, v) + tuple(inputs[3:])  # Returns the modified inputs and any additional parameters.
    handle = module.register_forward_pre_hook(hook)
    return handle


@torch.no_grad()
def compute_fisher_info(grads):
    fisher_info = grads.pow(2).sum(dim=0)
    return fisher_info


@torch.no_grad()
def compute_fisher_info_taylor1(grads):
    '''
    taylor1Scorer, method 6 from ICLR2017 paper
    <Pruning convolutional neural networks for resource efficient transfer learning>
    '''
    print("Start using taylor1 to calculate Fisher information...")
    if len(grads[0].shape) == 4:
        fisher_info1 = grads[0].abs().mean(-1).mean(-1).mean(0)
        # fisher_info = grads.pow(2).sum(dim=0)
    else:
        fisher_info1 = grads[0].abs().mean(1).mean(0)

    return fisher_info1