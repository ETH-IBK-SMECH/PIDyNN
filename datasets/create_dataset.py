from datasets.single_dof_duffing_oscillator import Duffing1DOFOscillator


def create_dataset(dataset_type, n_batches):
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
            't_end': 10.0,
            'dt': 0.01,
        }
        dataset = Duffing1DOFOscillator(example_system, example_parameters, n_batches=n_batches)
    else:
        raise NotImplemented

    return dataset
