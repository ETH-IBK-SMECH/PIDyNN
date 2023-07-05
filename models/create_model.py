import torch.nn as nn


def create_model(
        model_type,
        input_dim,
        hidden_dim,
        output_dim,
):

    if model_type == 'MLP':
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    else:
        raise NotImplementedError

    return model
