from datasets.single_dof_duffing_oscillator import Duffing1DOFOscillator
from datasets.mdof_nonlinear_oscillator import DuffingMDOFOscillator
import datasets.mdof_sim as mdof_sim
import numpy as np
from typing import Union


def create_dataset(phys_config: dict, sequence_length: int) -> Union[Duffing1DOFOscillator, None]:
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
    if phys_config['system-type'] == 'single_dof_duffing':
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
            'dt': 0.1,
        }
        dataset = Duffing1DOFOscillator(example_system, example_parameters, seq_len=sequence_length)
    elif phys_config['system-type'] == 'multi_dof_duffing':
        example_system = {
            'n_dof' : phys_config['n_dof'],
            'mass_vector' : np.array([1.0]*phys_config['n_dof']),
            'damping_vector' : np.array([0.25]*phys_config['n_dof']),
            'stiffness_vector' : np.array([10.0]*phys_config['n_dof']),
            'nonlinear_stiffness_vector' : np.array([2.0]+[0.0]*(phys_config['n_dof']-1)),
            'excitations' : [None]*(phys_config['n_dof']-2) + [   # example adds forcing at the N-1th DOF
                mdof_sim.actuators.rand_phase_ms(
                    freqs = np.array([0.7, 0.85, 1.6, 1.8]),
                    Sx = np.ones(4)
                )] + [None],
            'initial_conditions' : np.array([-2.0] + [0.0]*(phys_config['n_dof']-1) + [-2.0] + [0.0]*(phys_config['n_dof']-1))
        }
        example_parameters = {
            't_start': 0.0,
            't_end': 100.0,
            'dt': 0.1,
        }
        dataset = DuffingMDOFOscillator(example_system, example_parameters, seq_len=sequence_length)
    else:
        raise NotImplementedError("Specified dataset type is not implemented.")

    return dataset
