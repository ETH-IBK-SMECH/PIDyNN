import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datasets.create_dataset import create_dataset
from models.create_model import create_model


def main(config: argparse.Namespace) -> int:
    torch.manual_seed(42)
    device = torch.device('cpu')

    # Create dataset
    phases = ['train', 'val', 'test']
    if config.n_dof == 1:
        full_dataset = create_dataset('single_dof_duffing', config.sequence_length)
    else:
        full_dataset = create_dataset('multi_dof_duffing', config.sequence_length)
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
        x: DataLoader(dataset=datasets[x], batch_size=config.batch_size, shuffle=True if x == 'train' else False,
                      num_workers=config.num_workers, pin_memory=True) for x in
        phases}

    # Create model
    model = create_model(config.model_type, config.in_channels, config.latent_features, config.out_channels, config.sequence_length)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training loop
    epoch = 0
    model = model.to(device)
    while epoch < config.num_epochs:
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
                inputs = sample[..., 6:].to(device).float()
                targets = sample[..., :6].to(device).float()
                if phase == 'train':
                    optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                phase_loss += loss.item()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            print(('\tLoss {}'.format(phase_loss)))
        epoch += 1
    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="Train PINN")

    # physical-model arguments
    parser.add_argument('--n_dofs', type=int, default=4)
    parser.add_argument('--system-type', type=str, default='cantilever')

    # nn-model arguments
    parser.add_argument('--model-type', type=str, default='MLP')
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--latent-features', type=int, default=5)
    parser.add_argument('--out-channels', type=int, default=6)

    # training arguments
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--sequence-length', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    args = parser.parse_args()

    main(args)
