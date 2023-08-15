import numpy as np
import mdof_sim
from torch.utils.data import Dataset as BaseDataset


class DuffingMDOFOscillator(BaseDataset):
    def __init__(self, dynamic_system: dict, simulation_parameters: dict, data_params: dict):
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

        # set data parameters
        self.seq_len = data_params['seq_len']
        self.subsample = data_params['subsample']  # subsamples simulation data
        self.downsample = data_params['downsample']  # downsamples data for NN

        # normalize data
        self.maximum = data.max(axis=0)
        self.minimum = data.min(axis=0)
        self.alphas = self.maximum - self.minimum
        self.alphas[self.alphas==0.0] = 1e12  # to remove division by zero
        data = (data) / (self.alphas)

        # organise data
        # save ground_truth prior
        self.ground_truth = data
        coll_data = data[:(data.shape[0] // (self.subsample * self.seq_len)) * (self.subsample * self.seq_len)]  # cut off excess from subsampling and sequence ordering
        # subsample (this emulates lower sampling rate)
        data = coll_data[::self.subsample]
        coll_data = coll_data.reshape(-1, self.subsample, self.seq_len, 3 * n_dof + 1)
        self.coll_data = coll_data

        data = data.reshape(-1, self.seq_len, 3 * n_dof + 1)
        self.data = data

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index, ::self.downsample], self.coll_data[index]

    def get_original(self, index: int) -> np.ndarray:
        return self.ground_truth[index]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return self.__class__.__name__


if __name__ == '__main__':
    example_system: dict = {
        'n_dof': 3,
        'mass_vector': np.array([1.0] * 3),
        'damping_vector': np.array([0.25] * 3),
        'stiffness_vector': np.array([10.0] * 3),
        'nonlinear_stiffness_vector': np.array([20.0, 0.0, 0.0]),
        'excitations': [
            mdof_sim.actuators.rand_phase_ms(
                freqs=np.array([0.7, 0.85, 1.6, 1.8]),
                Sx=np.ones(4)
            ), None, None],
        'initial_conditions': None
    }

    example_parameters: dict = {
        't_start': 0.0,
        't_end': 60.0,
        'dt': 60 / 1024
    }

    data_parameters = {
        'seq_len' : 1,
        'subsample' : 4,
        'downsample' : 1
    }

    dataset = DuffingMDOFOscillator(example_system, example_parameters, data_parameters)

    u_data = dataset.data[..., :example_system['n_dof']].squeeze()
    ud_data = dataset.data[..., example_system['n_dof']:2 * example_system['n_dof']].squeeze()
    t_data = dataset.data[..., 2 * example_system['n_dof']].reshape(-1)
    u_coll = dataset.coll_data[..., :example_system['n_dof']].reshape(-1,example_system['n_dof'])
    ud_coll = dataset.coll_data[..., example_system['n_dof']:2 * example_system['n_dof']].reshape(-1,example_system['n_dof'])
    t_coll = dataset.coll_data[..., 2 * example_system['n_dof']].reshape(-1)
    f_coll = dataset.coll_data[..., 2 * example_system['n_dof'] + 1:].reshape(-1,example_system['n_dof'])

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, example_system['n_dof'], figsize=(12, 8))

    for i in range(example_system['n_dof']):
        axs[0,i].plot(t_data, u_data[:,i])
        axs[0,i].scatter(t_coll, u_coll[:,i], marker='x', color='tab:orange')
        axs[1,i].plot(t_data, ud_data[:,i])
        axs[1,i].scatter(t_coll, ud_coll[:,i], marker='x', color='tab:orange')
        axs[2,i].scatter(t_coll, f_coll[:,i], marker='x', color='tab:orange')

    plt.show()
