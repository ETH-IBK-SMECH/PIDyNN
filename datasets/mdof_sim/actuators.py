import numpy as np
import scipy
from math import pi

class excitation():

    def generate(self, time: np.ndarray):
        return self._generate(time)
    
class sinusoid(excitation):
    '''
    Single sinusoidal signal with central frequency w, amplitude f0, and phase phi
    '''

    def __init__(self, w: float, f0: float=1.0, phi: float=0.0):
        self.w = w
        self.f0 = f0
        self.phi = phi

    def _generate(self, tt: np.ndarray, seed: int=43810):
        return self.f0 * np.sin(self.w*tt + self.phi)
    
class white_gaussian(excitation):
    '''
    White Gaussian noise with variance f0, and mean
    '''

    def __init__(self, f0: float, mean: float=0.0):
        self.f0 = f0
        self.u = mean
    
    def _generate(self, tt: np.ndarray, seed: int=43810):
        ns = tt.shape[0]
        np.random.seed(seed)
        return np.random.normal(self.u, self.f0*np.ones((ns)))
    
class sine_sweep(excitation):
    '''
    Sine sweep signal
    '''

    def __init__(self, w_l: float, w_u: float, f0: float=1.0, scale: str='linear'):
        self.w_l = w_l
        self.w_u = w_u
        self.f0 = f0
        self.scale = scale

    def _generate(self, tt: np.ndarray, seed: int=43810):
        f0 = self.w_l / (2*pi)
        f1 = self.w_u / (2*pi)
        F =  f0 * scipy.signal.chirp(tt, f0, tt[-1], f1, method=self.scale)
        return F
    
class rand_phase_ms(excitation):
    '''
    Random-phase multi-sine
    '''

    def __init__(self, freqs: np.ndarray, Sx: np.ndarray):
        self.freqs = freqs
        self.Sx = Sx

    def _generate(self, tt: np.ndarray, seed: int=43810):

        np.random.seed(seed)
        phases = np.random.rand(self.freqs.shape[0])
        F_mat = np.sin(tt.reshape(-1,1) @ self.freqs.reshape(1,-1) + phases.T)

        return (F_mat @ self.Sx).reshape(-1)

class shaker():
    '''
    Shaker class generates force signals at each DOF using excitation class
    '''

    def __init__(self, excitations=None, seed: int=43810):

        self.excitations = excitations
        self.dofs = len(excitations)
        self.seed = seed

    def generate(self, tt: np.ndarray):
        nt = tt.shape[0]
        self.f = np.zeros((self.dofs,nt))
        for n, excite in enumerate(self.excitations):
            match excite:
                case excitation():
                    self.f[n,:] = self.excitations[n]._generate(tt, self.seed+n)
                case np.ndarray():
                    self.f[n,:] = self.excitations[n]
                case None:
                    self.f[n,:] = np.zeros(nt)

        return self.f
        


