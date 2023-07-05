import torch.nn as nn
from typing import Union


def create_model(
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int
) -> nn.Module:
    """
    Create a model based on the given model type.

    Args:
        model_type: Type of model.
        input_dim: Dimensionality of the input.
        hidden_dim: Dimensionality of the hidden layers.
        output_dim: Dimensionality of the output.
        seq_len: Length of sequences.

    Returns:
        The created model.

    Raises:
        NotImplementedError: If the specified model type is not implemented.
    """
    if model_type == 'MLP':
        model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim * seq_len, hidden_dim * seq_len),
            nn.Tanh(),
            nn.Linear(hidden_dim * seq_len, hidden_dim * seq_len),
            nn.Tanh(),
            nn.Linear(hidden_dim * seq_len, output_dim * seq_len),
            nn.Unflatten(dim=1, unflattened_size=(seq_len, output_dim)),
        )
    else:
        raise NotImplementedError("Specified model type is not implemented.")

    return model
