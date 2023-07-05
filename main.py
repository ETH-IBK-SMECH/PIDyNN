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

    # create dataset
    phases = ['train', 'test']
    full_dataset = create_dataset('single_dof_duffing', batch_size * 10)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    datasets = {
        'train': train_dataset,
        'test': test_dataset
    }
    dataloaders = {
        x: DataLoader(dataset=datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False,
                      num_workers=num_workers, pin_memory=True) for x in
        phases}

    # Training loop
    epoch = 0
    while epoch < num_epochs:
        print('Epoch {}'.format(epoch))
        for phase in ['train', 'test']:

            for i, sample in tqdm(enumerate(dataloaders[phase]),
                                  total=int(len(datasets[phase]) / dataloaders[phase].batch_size)):
                sample = sample.to(device)
                #print(sample)

        epoch += 1



    # TODO create model


    return 0


if __name__ == '__main__':
    main()
