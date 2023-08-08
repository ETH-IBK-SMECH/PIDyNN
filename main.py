import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datasets.create_dataset import create_dataset
from models.create_model import create_model
from models.pinn_models import ParamClipper
from plotter.plotter import Plotter


def main(config: argparse.Namespace) -> int:
    torch.manual_seed(42)
    device = torch.device('cpu')

    # Create dataset
    phys_config = {
        'n_dof': config.n_dof,
        'system-type': config.system_type
    }
    phases = ['train', 'val', 'test']
    full_dataset = create_dataset(phys_config, config.sequence_length)
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                             [train_size, val_size, test_size])
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    dataloaders = {
        x: DataLoader(dataset=datasets[x], batch_size=config.batch_size, shuffle=True if x == 'train' else False,
                      num_workers=config.num_workers, pin_memory=True) for x in phases}

    # Create model
    if config.out_channels != 2 * config.n_dof:
        raise Exception("Number of network outputs does not match state vector of simulated model")
    pinn_config = {
        'n_dof': config.n_dof,
        'phys_system_type': config.phys_system_type,
        'system_discovery': config.system_discovery,
        'm_': config.m_,
        'c_': config.c_,
        'k_': config.k_,
        'kn_': config.kn_,
        'alphas': None,
        'param_norms': {
            'm': 1.0,
            'c': 1.0,
            'k': 10.0,
            'kn': 10.0
        }
    }
    model = create_model(config.model_type, config.in_channels, config.latent_features, config.out_channels,
                         config.sequence_length, pinn_config)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training loop
    epoch = 0
    model = model.to(device)
    progress_bar = tqdm(total=config.num_epochs)  # progress bar moved outward to preserve while loop structure
    while epoch < config.num_epochs:
        write_string = ''
        write_string += 'Epoch {}\n'.format(epoch)
        for phase in phases:
            phase_loss = 0.
            write_string += '\tPhase {}\n'.format(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for i, sample in enumerate(dataloaders[phase]):
                # parse data sample
                state = sample[..., :2 * config.n_dof].to(device).float()
                force = sample[..., 2 * config.n_dof + 1:].to(device).float()
                t_span = sample[:, :, 2 * config.n_dof].to(device).float()

                # unnormalize time component
                t_span = t_span * (datasets[phase].dataset.maximum[2 * config.n_dof] - datasets[phase].dataset.minimum[2 * config.n_dof]) + datasets[phase].dataset.minimum[2 * config.n_dof]

                # switch according to task
                match config.task:
                    case 'regression':
                        inputs = t_span
                        targets = state
                    case 'k_plus_1':
                        inputs = (state[:, :-1], t_span[0, :-1] - t_span[0, :-1].min())  # ODE only needs relative time, possible work-arounds here: https://github.com/rtqichen/torchdiffeq/issues/122
                        targets = state[:, 1:]
                    case 'pgnn':
                        inputs = state[:, :-1]
                        targets = state[:, 1:]
                    case _:
                        raise NotImplementedError

                if phase == 'train':
                    optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)

                phase_loss += loss.item()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            write_string += '\tLoss {}\n'.format(phase_loss)

        # Uncomment this line if you want to keep track of the loss (preferably deactivate progress bar before that)
        #print(write_string)
        epoch += 1
        progress_bar.update(1)

    progress_bar.close()

    # plot results
    model.eval()

    # plot last sample of dataset
    sample = datasets['test'][-1]

    ground_truth = datasets['test'].dataset.get_original(datasets['test'].indices[-1])
    inputs = torch.from_numpy(sample[:-1, :2 * config.n_dof]).to(device).float().unsqueeze(0)
    t_span = torch.from_numpy(sample[:, 2 * config.n_dof]).to(device).float().unsqueeze(0)
    ground_t_span = torch.from_numpy(ground_truth[:, 2 * config.n_dof]).to(device).float().unsqueeze(0)
    t_span = t_span * (
            datasets['test'].dataset.maximum[2 * config.n_dof] - datasets['test'].dataset.minimum[2 * config.n_dof]) + \
             datasets['test'].dataset.minimum[2 * config.n_dof]
    ground_t_span = ground_t_span * (
            datasets['test'].dataset.maximum[2 * config.n_dof] - datasets['test'].dataset.minimum[2 * config.n_dof]) + \
             datasets['test'].dataset.minimum[2 * config.n_dof]
    t_span = t_span[0] - t_span[0].min()
    ground_t_span = ground_t_span[0] - ground_t_span[0].min()
    match config.task:
        case 'regression' | 'k_plus_1':
            input_tensor = [inputs, t_span]
        case 'pgnn':
            input_tensor = inputs
    predictions = model(input_tensor).detach().cpu().squeeze().numpy()

    pinn_plotter = Plotter(config.textwidth, config.fontsize, config.fontname)
    test_figure = pinn_plotter.plot_predictions(config.n_dof, sample, predictions, ground_truth, t_span, ground_t_span)
    pinn_plotter.show_figure()

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="Train PINN")

    # physical-model arguments
    parser.add_argument('--n-dof', type=int, default=1)
    parser.add_argument('--system-type', type=str, default='single_dof_duffing')

    # nn-model arguments
    parser.add_argument('--model-type', type=str, default='PGNN')
    parser.add_argument('--in-channels', type=int, default=2)
    parser.add_argument('--latent-features', type=int, default=32)
    parser.add_argument('--out-channels', type=int, default=2)

    # pinn arguments
    parser.add_argument('--phys-system-type', type=str, default='duffing_sdof')
    parser.add_argument('--lambdas', type=dict, default={
        'obs': 5.0,
        'cc': 10.0,
        'ode': 5.0
    })
    parser.add_argument('--system-discovery', type=bool, default=True)
    parser.add_argument('--m-', type=torch.Tensor, default=torch.Tensor([[10.0]]))
    parser.add_argument('--c-', type=torch.Tensor, default=torch.Tensor([[1.0]]))
    parser.add_argument('--k-', type=torch.Tensor, default=torch.Tensor([[15.0]]))
    parser.add_argument('--kn-', type=torch.Tensor, default=torch.Tensor([[100.0]]))

    # training arguments
    parser.add_argument('--task', type=str, default="pgnn")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--sequence-length', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # plotting arguments
    parser.add_argument('--textwidth', type=float, default=32.0)
    parser.add_argument('--fontsize', type=int, default=10)
    parser.add_argument('--fontname', type=str, default="times")

    args = parser.parse_args()

    main(args)
