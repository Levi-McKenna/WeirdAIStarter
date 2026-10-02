"""
Custom collate function for Weird AI instruction fine-tuning.
"""

import torch


def custom_collate_fn(
    batch,
    pad_token_id=0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Pad variable-length token ID lists, create shifted targets, and mask extra padding.
    """

    # TODO:
    # Find the maximum length in the batch.
    # Add 1 because we append a padding/end token before creating targets.
    batch_max_length = max(map(len, batch)) + 1

    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()

         # TODO: append one pad_token_id to new_item
        new_item += [pad_token_id]

        # TODO: pad new_item to batch_max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # TODO: inputs are padded[:-1], targets are padded[1:]
        inputs = padded[:-1]
        targets = padded[1:]

        # TODO: replace all but the first pad_token_id in targets with ignore_index
        first_pad = targets.index(pad_token_id)
        targets[first_pad + 1:] = [ignore_index] * (len(targets) - (first_pad + 1))

        # TODO: optionally truncate inputs and targets to allowed_max_length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(torch.tensor(inputs))
        targets_lst.append(torch.tensor(targets))

    # TODO: stack input and target tensors, then move them to device.
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
