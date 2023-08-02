import numpy as np
import datasets.mdof_sim as mdof_sim
from torch.utils.data import Dataset as BaseDataset


class DuffingMDOFOscillator(BaseDataset):
    def __init__(self, dynamic_system: dict, simulation_parameters: dict, seq_len: int):
        print('Simulating MDOF Duffing oscillator')

        n_dof = dynamic_system['n_dof']
        t_span = np.arange(simulation_parameters['t_start'], simulation_parameters['t_end'],
                           simulation_parameters['dt'])
        m_ = dynamic_system['mass_vector']
        c_ = dynamic_system['damping_vector']
        k_ = dynamic_system['stiffness_vector']
        kn_ = dynamic_system['nonlinear_stiffness_vector']

        # simulate using mdof_sim package
        # create nonlinearity
        cubic_nonlin = mdof_sim.nonlinearities.exponent_stiffness(kn_, exponent=3, dofs=n_dof)
        # instantiate system
        system = mdof_sim.systems.cantilever(m_, c_, k_, dofs=n_dof, nonlinearity=cubic_nonlin)
        # generate excitations
        system.excitations = dynamic_system['excitations']
        solution = system.simulate(t_span, z0=dynamic_system['initial_conditions'])

        # add time and forcing to dataset
        data = np.concatenate((solution, t_span.reshape(-1,1), system.f.T),axis=1)

        # normalize data
        self.maximum = data.max(axis=0)
        self.minimum = data.min(axis=0)
        self.alphas = self.maximum - self.minimum
        self.alphas[self.alphas==0.0] = 1e12  # to remove division by zero
        #data = (data) / (self.alphas)
        data = (data - self.minimum) / (self.maximum - self.minimum)
        self.downsample = simulation_parameters['downsample']

        # reshape to number of batches
        # 2 n_dof for state, 1 n_dof for force, and 1 for time
        data = np.reshape(data, [-1, seq_len*self.downsample, 3 * n_dof + 1])
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
    example_system: dict = {
        'n_dof': 4,
        'mass_vector': np.array([1.0] * 4),
        'damping_vector': np.array([0.25] * 4),
        'stiffness_vector': np.array([10.0] * 4),
        'nonlinear_stiffness_vector': np.array([2.0, 0.0, 0.0, 0.0]),
        'excitations': [
            None,
            None,
            mdof_sim.actuators.rand_phase_ms(
                freqs=np.array([0.7, 0.85, 1.6, 1.8]),
                Sx=np.ones(4)
            ),
            None],
        'initial_conditions': np.array([-2.0, 0.0, 0.0, 3.0, -2.0, 0.0, 0.0, 0.0])
    }

    example_parameters: dict = {
        't_start': 0.0,
        't_end': 1200.0,
        'dt': 0.01,
        'downsample': 10,
    }

    dataset = DuffingMDOFOscillator(example_system, example_parameters, seq_len=1200)

    sample = dataset[-1]
    ground_truth = dataset.get_original(-1)
    import matplotlib.pyplot as plt

    plt.plot(sample[:, -1], sample[:, 0], 'o')
    plt.plot(ground_truth[:, -1], ground_truth[:, 0])
    plt.show()
