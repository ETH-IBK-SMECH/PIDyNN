from datasets.single_dof_duffing_oscillator import Duffing1DOFOscillator
from typing import Union


def create_dataset(dataset_type: str, sequence_length: int) -> Union[Duffing1DOFOscillator, None]:
    """
    Create a dataset based on the given dataset type.

    Args:
        dataset_type: Type of dataset.
        sequence_length: Length of sequences in the dataset.

    Returns:
        The created dataset.

    Raises:
        NotImplementedError: If the specified dataset type is not implemented.
    """
    if dataset_type == 'single_dof_duffing':
        example_system = {
            'mass': 1.0,
            'stiffness': 1.0,
            'damping': 0.1,
            'nonlinear_stiffness': 0.5,
            'initial_conditions': [0.0, 0.0],
        }
        example_parameters = {
            't_start': 0.0,
            't_end': 100.0,
            'dt': 0.01,
        }
        dataset = Duffing1DOFOscillator(example_system, example_parameters, seq_len=sequence_length)
    else:
        raise NotImplementedError("Specified dataset type is not implemented.")

    return dataset
