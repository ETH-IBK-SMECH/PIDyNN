import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets.create_dataset import create_dataset


def main():
    # args
    device = torch.device('cpu')
    batch_size = 10
    num_workers = 2
    num_epochs = 10

    torch.manual_seed(42)

    # Create dataset
    phases = ['train', 'val', 'test']
    full_dataset = create_dataset('single_dof_duffing', batch_size * 10)
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

    # Training loop
    epoch = 0
    while epoch < num_epochs:
        print('Epoch {}'.format(epoch))
        for phase in phases:
            print('\tPhase {}'.format(phase))
            for i, sample in tqdm(enumerate(dataloaders[phase]),
                                  total=int(len(datasets[phase]) / dataloaders[phase].batch_size)):
                sample = sample.to(device)
                #print(sample)

        epoch += 1


    return 0


if __name__ == '__main__':
    main()
