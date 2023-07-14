import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datasets.create_dataset import create_dataset
from models.create_model import create_model
from plotter import Plotter


def main(config: argparse.Namespace) -> int:
    torch.manual_seed(42)
    device = torch.device('cpu')

    # Create dataset
    phys_config = {
        'n_dof' : config.n_dof,
        'system-type' : config.system_type
    }
    phases = ['train', 'val', 'test']
    full_dataset = create_dataset(phys_config, config.sequence_length)
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
    if config.out_channels != 2*config.n_dof:
        raise Exception("Number of network outputs does not match state vector of simulated model")
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
                inputs = sample[..., 2*config.n_dof:].to(device).float()
                targets = sample[..., :2*config.n_dof].to(device).float()
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

    # plot results
    model.eval()

    # plot last sample of dataset
    sample = datasets['test'][-1]
    ground_truth = datasets['test'].dataset.get_original(datasets['test'].indices[-1])
    inputs = torch.from_numpy(sample[..., 2 * config.n_dof:]).to(device).float().unsqueeze(0)
    predictions = model(inputs).detach().cpu().squeeze().numpy()

    pinn_plotter = Plotter()
    test_figure = pinn_plotter.plot_predictions(config.n_dof, sample, predictions, ground_truth)
    pinn_plotter.show_figure()

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="Train PINN")

    # physical-model arguments
    parser.add_argument('--n-dof', type=int, default=4)
    parser.add_argument('--system-type', type=str, default='multi_dof_duffing')

    # nn-model arguments
    parser.add_argument('--model-type', type=str, default='MLP')
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--latent-features', type=int, default=5)
    parser.add_argument('--out-channels', type=int, default=8)

    # training arguments
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--sequence-length', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    args = parser.parse_args()

    main(args)
