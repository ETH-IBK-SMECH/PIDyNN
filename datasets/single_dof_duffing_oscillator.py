import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from torch.utils.data import Dataset as BaseDataset

"""
Remove this file once MDOF duffing oscillator is created.
"""


def duffing_oscillator(y, t, f, k, c, alpha, m):
    x, v = y  # Unpack the state variables

    dxdt = v
    dvdt = (f(t) - c * v - k * x - alpha * x ** 3) / m

    return [dxdt, dvdt]


class Duffing1DOFOscillator(BaseDataset):

    def __init__(self,
                 dynamic_system,
                 simulation_parameters,
                 seq_len
                 ):

        print('Simulating 1DOF Duffing oscillator...')

        n_dof = 1
        t_span = np.arange(simulation_parameters['t_start'], simulation_parameters['t_end'], simulation_parameters['dt'])
        external_force = np.random.normal(0, 1, [len(t_span), 1])
        fint = interp1d(t_span, external_force[:, 0], fill_value='extrapolate')

        # Integrate the system using odeint
        solution = odeint(duffing_oscillator,
                          dynamic_system['initial_conditions'],
                          t_span,
                          args=(
                              fint,
                              dynamic_system['stiffness'],
                              dynamic_system['damping'],
                              dynamic_system['nonlinear_stiffness'],
                              dynamic_system['mass']
                          )
                          )

        # add forcing to dataset
        data = np.concatenate([solution, external_force], axis=1)

        # normalize data
        self.maximum = data.max(axis=0)
        self.minimum = data.min(axis=0)
        data = (data - self.minimum) / (self.maximum - self.minimum)

        # reshape to number of batches
        # 2 n_dof for state and 1 n_dof for forces
        data = np.reshape(data, [-1, seq_len, 3*n_dof])

        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    np.random.seed(42)

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

    dataset = Duffing1DOFOscillator(example_system, example_parameters, seq_len=100)

    sample = dataset[-1]

    import matplotlib.pyplot as plt
    plt.plot(sample)
    plt.show()
