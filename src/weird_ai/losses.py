"""
Loss utilities for Weird AI.

These functions calculate how well the model predicts the next token.
"""

import torch
import torch.nn.functional as F


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate cross-entropy loss for one batch.

    Args:
        input_batch: Tensor of shape (batch_size, num_tokens).
        target_batch: Tensor of shape (batch_size, num_tokens).
        model: The Weird AI model.
        device: CPU or CUDA device.

    Returns:
        A scalar loss tensor.
    """

    # TODO:
    # 1. Move input_batch and target_batch to the selected device.
    # 2. Run input_batch through the model to get logits.
    # 3. Reshape logits so cross_entropy sees:
    #       (batch_size * num_tokens, vocab_size)
    # 4. Reshape targets so cross_entropy sees:
    #       (batch_size * num_tokens)
    # 5. Return cross-entropy loss.

    input_batch.to(device)
    target_batch.to(device)

    logits = model(input_batch)

    return F.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss across a data loader.

    Args:
        data_loader: A PyTorch DataLoader.
        model: The Weird AI model.
        device: CPU or CUDA device.
        num_batches: Optional limit on number of batches to evaluate.

    Returns:
        Average loss as a float.
    """

    # TODO:
    # 1. Handle an empty data loader.
    # 2. Determine how many batches to evaluate.
    # 3. Loop through the data loader.
    # 4. Calculate loss for each batch.
    # 5. Return the average loss.

    total_loss = 0
    if len(data_loader) == 0:
        return float("NaN")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        batch_loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += batch_loss.item()

    return total_loss / num_batches


def calculate_perplexity(loss):
    """
    Convert cross-entropy loss into perplexity.

    Args:
        loss: A scalar loss value or tensor.

    Returns:
        Perplexity value.
    """

    # TODO:
    # Perplexity is exp(loss).
    return torch.exp(torch.tensor(loss))
