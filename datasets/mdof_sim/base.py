import numpy as np
from datasets.mdof_sim.simulators import *
from datasets.mdof_sim.actuators import shaker
import warnings

class mdof_system:
    '''
    Base class for generic mdof system
    '''

    def __init__(self, M=None, C=None, K=None, Cn=None, Kn=None):

        self.M = M
        self.C = C
        self.K = K
        self.Cn = Cn
        self.Kn = Kn
        self.dofs = M.shape[0]

        self.gen_state_matrices()

    def gen_state_matrices(self):
        '''
        Generate state matrices A, H, and nonlinear matrix An based on system parameters
        '''

        self.A = np.concatenate((
            np.concatenate((np.zeros((self.dofs,self.dofs)), np.eye(self.dofs)), axis=1),
            np.concatenate((-np.linalg.inv(self.M)@self.K, -np.linalg.inv(self.M)@self.C), axis=1)
        ), axis=0)

        self.H = np.concatenate((np.zeros((self.dofs, self.dofs)), np.linalg.inv(self.M)), axis=0)

        match [self.Cn, self.Kn]:
            case [None,None]:
                self.An = None
            case [_,None]:
                self.An = np.concatenate((
                    np.concatenate((np.zeros((self.dofs,self.dofs)), np.eye(self.dofs)), axis=1),
                    np.concatenate((np.zeros_like(self.Cn), -np.linalg.inv(self.M)@self.Cn), axis=1)
                ), axis=0)
            case [None, _]:
                self.An = np.concatenate((
                    np.concatenate((np.zeros((self.dofs,self.dofs)), np.eye(self.dofs)), axis=1),
                    np.concatenate((-np.linalg.inv(self.M)@self.Kn, np.zeros_like(self.Kn)), axis=1)
                ), axis=0)
            case [_,_]:
                self.An = np.concatenate((
                    np.concatenate((np.zeros((self.dofs,self.dofs)), np.eye(self.dofs)), axis=1),
                    np.concatenate((-np.linalg.inv(self.M)@self.Kn, -np.linalg.inv(self.M)@self.Cn), axis=1)
                ), axis=0)

    def simulate(self, tt, z0=None, simulator=None):
        '''
        Simulate the system for a given time using the specified simulator.

        Args:
            tt: The vector of time samples
            z0: The initial state of the system. Defaults to None.
            simulator: The simulator to use for the simulation. Defaults to scipy.solve_ivp.

        Returns:
            A dictionary containing the system's response displacement and velocity over time.
        '''

        # instate time
        self.t = tt

        if hasattr(self, 'excitations'):
            # create shaker object
            self.shaker = shaker(self.excitations)
            # generate forcing series
            self.f = self.shaker.generate(tt)
        else:
            self.f = None

        # initiate simulator
        if simulator is None:
            self.simulator = scipy_ivp(self)
        else:
            self.simulator = simulator(self)

        # initial conditions
        if z0 is None:
            warnings.warn('No initial conditions provided, proceeding with zero initial state', UserWarning)
            z0 = np.zeros((2*self.dofs))
            if all([e is None for e in self.excitations]):
                warnings.warn('Zero initial condition and zero excitations, what do you want??', UserWarning)
        
        # simulate
        return self.simulator.sim(tt, z0).T  # transpose to make it consistent with pytorch





        