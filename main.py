import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets.create_dataset import create_dataset
from models.create_model import create_model


def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    # args
    batch_size = 10
    sequence_length = 100
    num_workers = 2
    num_epochs = 10
    learning_rate = 1e-3
    weight_decay = 1e-4
    model_type = 'MLP'
    in_channels = 1
    latent_features = 5
    output_channels = 2

    # Create dataset
    phases = ['train', 'val', 'test']
    full_dataset = create_dataset('single_dof_duffing', sequence_length)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = int(0.1 * len(full_dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    dataloaders = {
        x: DataLoader(dataset=datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False,
                      num_workers=num_workers, pin_memory=True) for x in
        phases}

    # Create model
    model = create_model(model_type, in_channels, latent_features, output_channels, sequence_length)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    epoch = 0
    model = model.to(device)
    while epoch < num_epochs:
        print('Epoch {}'.format(epoch))
        for phase in phases:

            phase_loss = 0.

            print('\tPhase {}'.format(phase))
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for i, sample in tqdm(enumerate(dataloaders[phase]),
                                  total=int(len(datasets[phase]) / dataloaders[phase].batch_size)):

                # This data parsing is specific to the dummy example and will have to be changed
                inputs = sample[..., 2:].to(device).float()
                targets = sample[..., :2].to(device).float()

                if phase == 'train':
                    optimizer.zero_grad()

                predictions = model(inputs)
                loss = criterion(predictions,  targets)
                phase_loss += loss.item()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            print(('\tLoss {}'.format(phase_loss)))

        epoch += 1

    return 0


if __name__ == '__main__':
    main()
