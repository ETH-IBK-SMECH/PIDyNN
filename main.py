import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from tqdm.auto import tqdm as tqdma
from datasets.create_dataset import create_dataset
from models.create_model import create_model
from models.pinn_models import ParamClipper
from plotter.plotter import Plotter

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
    train_size = int(1.0 * len(full_dataset))  # training size in number of batches
    val_size = int(0 * len(full_dataset))  # validation size in number of batches
    test_size = int(0 * len(full_dataset))  # testing size in number of batches
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    dataloaders = {
        x: DataLoader(dataset=datasets[x], batch_size=config.batch_size, shuffle=True if x == 'train' else False, num_workers=config.num_workers, pin_memory=True) for x in phases}

    # Create model
    if config.out_channels != 2*config.n_dof:
        raise Exception("Number of network outputs does not match state vector of simulated model")
    pinn_config = {
        'n_dof' : config.n_dof,
        'phys_system_type' : config.phys_system_type,
        'system_discovery' : config.system_discovery,
        'm_' : config.m_,
        'c_' : config.c_,
        'k_' : config.k_,
        'kn_' : config.kn_,
        'alphas' : full_dataset.alphas,
        'param_norms' : {
            'm' : 1.0,
            'c' : 1.0,
            'k' : 10.0,
            'kn' : 10.0
        }
    }
    model = create_model(config.model_type, config.in_channels, config.latent_features, config.out_channels, config.sequence_length, pinn_config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.set_switches(config.lambdas)
    clipper = ParamClipper()

    loss_hist = []
    # Training loop
    epoch = 0
    model = model.to(device)
    for epoch in tqdm(range(config.num_epochs)):
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
                targets = sample[..., :2*config.n_dof].to(device).float().requires_grad_()
                inputs = sample[..., 2*config.n_dof].to(device).float().requires_grad_()
                force = sample[..., 2*config.n_dof+1:].to(device).float().requires_grad_()
                if phase == 'train':
                    optimizer.zero_grad()
                loss, losses, _ = model.loss_func(config.lambdas, inputs, targets, inputs, force)
                phase_loss += loss.item()
                if phase == 'train':
                    predictions = model(inputs)
                    loss.backward()
                    optimizer.step()
                    model.apply(clipper)
                    loss_hist.append([loss_it.item() for loss_it in losses] + [loss.item()])
            if phase == 'train':
                if config.system_discovery:
                    write_string += '\tSystem Parameters:\tc - {:.4f} [{:.2f}]\tk - {:.4f} [{:.2f}]\tkn - {:.4f} [{:.2f}]\n'.format(
                        model.c_[0,0].item()*pinn_config['param_norms']['c'],
                        config.c_[0,0].item(),
                        model.k_[0,0].item()*pinn_config['param_norms']['k'],
                        config.k_[0,0].item(),
                        model.kn_[0,0].item()*pinn_config['param_norms']['kn'],
                        config.kn_[0,0].item())
            
            write_string += '\tLoss {}\n'.format(phase_loss)
            tqdma.write(write_string)

    # plot results
    model.eval()

    # plot all samples of dataset
    # generate and collate all samples to full time window
    inputs_all = torch.zeros((config.batch_size * config.sequence_length, 1)).numpy()
    predictions_all = torch.zeros((config.batch_size * config.sequence_length, 2 * config.n_dof)).numpy()
    gt_all = torch.zeros((config.batch_size * config.sequence_length, 3 * config.n_dof + 1)).numpy()
    samples_all = torch.zeros((config.batch_size * config.sequence_length, 3 * config.n_dof + 1)).numpy()
    for i, sample in enumerate(datasets['train']):
        samples_all[i*config.sequence_length:(i+1)*config.sequence_length] = sample
        gt_all[i*config.sequence_length:(i+1)*config.sequence_length,:] = datasets['train'].dataset.get_original(datasets['train'].indices[i])
        inputs = torch.from_numpy(sample[..., 2*config.n_dof]).to(device).float().unsqueeze(0)
        inputs_all[i*config.sequence_length:(i+1)*config.sequence_length] = inputs.T.numpy()
        predictions_all[i*config.sequence_length:(i+1)*config.sequence_length] = model(inputs).detach().cpu().squeeze().numpy()
    includes = {
        "gt" : config.include_gt,
        "pred" : config.include_pred,
        "obs" : config.include_obs
    }

    pinn_plotter = Plotter(config.textwidth, config.fontsize, config.fontname, config.fig_ratio, includes)
    test_figure = pinn_plotter.plot_predictions(config.n_dof, samples_all, predictions_all, gt_all)
    pinn_plotter.show_figure()

    return 0


if __name__ == '__main__':
    # parse config
    parser = argparse.ArgumentParser(description="Train PINN")

    # physical-model arguments
    parser.add_argument('--n-dof', type=int, default=1)
    parser.add_argument('--system-type', type=str, default='single_dof_duffing')

    # nn-model arguments
    parser.add_argument('--model-type', type=str, default='sdof_pinn')
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--latent-features', type=int, default=32)
    parser.add_argument('--out-channels', type=int, default=2)

    # pinn arguments
    parser.add_argument('--phys-system-type', type=str, default='duffing_sdof')
    parser.add_argument('--lambdas', type=dict, default={
        'obs' : 1.0,
        'cc' : 1.0,
        'ode' : 1.0
    })
    parser.add_argument('--system-discovery', type=bool, default=True)
    parser.add_argument('--m-', type=torch.Tensor, default=torch.Tensor([[10.0]]))
    parser.add_argument('--c-', type=torch.Tensor, default=torch.Tensor([[1.0]]))
    parser.add_argument('--k-', type=torch.Tensor, default=torch.Tensor([[15.0]]))
    parser.add_argument('--kn-', type=torch.Tensor, default=torch.Tensor([[100.0]]))

    # training arguments
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=200000)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # plotting arguments
    parser.add_argument('--textwidth', type=float, default=14.4)
    parser.add_argument('--fontsize', type=int, default=12)
    parser.add_argument('--fontname', type=str, default="cmunrm")
    parser.add_argument('--fig-ratio', type=float, default=12/16)
    parser.add_argument('--include-gt', type=bool, default=True)
    parser.add_argument('--include-pred', type=bool, default=True)
    parser.add_argument('--include-obs', type=bool, default=False)

    args = parser.parse_args()

    main(args)
