import torch.nn as nn


def create_model(
        model_type,
        input_dim,
        hidden_dim,
        output_dim,
        seq_len,
):

    if model_type == 'MLP':
        model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim*seq_len, hidden_dim*seq_len),
            nn.ReLU(),
            nn.Linear(hidden_dim*seq_len, hidden_dim*seq_len),
            nn.ReLU(),
            nn.Linear(hidden_dim*seq_len, output_dim*seq_len),
            nn.Unflatten(dim=1, unflattened_size=(seq_len, output_dim)),
        )
    else:
        raise NotImplementedError

    return model
