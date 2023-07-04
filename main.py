import torch

from datasets.create_dataset import create_dataset


def main():

    # create dataset
    full_dataset = create_dataset('single_dof_duffing', 10)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # TODO create model

    # TODO write training loop

    return 0


if __name__ == '__main__':
    main()

