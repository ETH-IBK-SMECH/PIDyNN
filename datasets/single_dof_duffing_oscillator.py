import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from torch.utils.data import Dataset as BaseDataset
from math import pi


def duffing_oscillator(y: np.ndarray, t: float, f, k: float, c: float, alpha: float, m: float) -> np.ndarray:
    """
    Defines the equations of motion for the Duffing oscillator.

    Args:
        y: Array of position and velocity.
        t: Time.
        f: External force function.
        k: Stiffness coefficient.
        c: Damping coefficient.
        alpha: Nonlinear coefficient.
        m: Mass of the oscillator.

    Returns:
        Array of derivatives [dxdt, dvdt].
    """
    x, v = y  # Unpack the state variables

    dxdt = v
    dvdt = (f(t) - c * v - k * x - alpha * x ** 3) / m

    return np.array([dxdt, dvdt])


class Duffing1DOFOscillator(BaseDataset):
    def __init__(self, dynamic_system: dict, simulation_parameters: dict, seq_len: int):
        print('Simulating 1DOF Duffing oscillator...')

        n_dof = 1
        t_span = np.arange(simulation_parameters['t_start'], simulation_parameters['t_end'],
                           simulation_parameters['dt'])
        external_force = np.random.normal(0, 1, [len(t_span), 1])
        freqs = np.array([0.7, 0.85, 1.6, 1.8])
        np.random.seed(43810)
        phases = np.random.rand(freqs.shape[0], 1)
        F_mat = np.sin(t_span.reshape(-1, 1) @ freqs.reshape(1, -1) + phases.T)
        Sx = np.ones((4, 1))
        external_force = F_mat @ Sx
        fint = interp1d(t_span, external_force[:, 0], fill_value='extrapolate')

        # Integrate the system using odeint
        solution = odeint(
            duffing_oscillator,
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
        data = np.concatenate([solution, t_span.reshape(-1, 1)], axis=1)

        # normalize data
        self.maximum = data.max(axis=0)
        self.minimum = data.min(axis=0)
        self.downsample = simulation_parameters['downsample']
        data = (data - self.minimum) / (self.maximum - self.minimum)

        # reshape to number of batches
        # 2 n_dof for state and 1 n_dof for time
        data = np.reshape(data, [-1, seq_len*self.downsample, 2 * n_dof + 1])
        self.data = data

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index, ::self.downsample]

    def get_original(self, index: int) -> np.ndarray:
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return self.__class__.__name__


if __name__ == '__main__':
    np.random.seed(42)

    example_system: dict = {
        'mass': 10.0,
        'stiffness': 15.0,
        'damping': 1.0,
        'nonlinear_stiffness': 100.0,
        'initial_conditions': [0.0, 0.0],
    }

    example_parameters: dict = {
        't_start': 0.0,
        't_end': 120.0,
        'dt': 120 / 1024,
        'downsample': 12,
    }

    dataset = Duffing1DOFOscillator(example_system, example_parameters, seq_len=128)

    x = dataset[:, :, 0].reshape(-1)
    v = dataset[:, :, 1].reshape(-1)
    t = dataset[:, :, 2].reshape(-1)
    f = dataset[:, :, 3].reshape(-1)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    axs[0].plot(t, x)
    axs[1].plot(t, v)
    axs[2].plot(t, f)
    plt.show()
