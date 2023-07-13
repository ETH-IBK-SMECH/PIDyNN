import torch

class nonlinearity():

    def __init__(self):
        pass

    def z_func(self, z):
        raise Exception('No nonlinearity selected')

class exponent_stiffness(nonlinearity):

    def __init__(self, exponent: int=3):
        self.exponent = exponent
        self.name = 'exponent_stiffness'

    def z_func(self, z):
        dofs = int(z.shape[0]/2)
        return z[:dofs]**self.exponent
    

class exponent_damping(nonlinearity):

    def __init__(self, exponent: int=2):
        self.exponent = exponent
        self.name = 'exponent_damping'

    def z_func(self, z):
        dofs = int(z.shape[0]/2)
        return z[:dofs]**self.exponent
    

class exponent_stiff_damp(nonlinearity):

    def __init__(self, stiff_exp: int=3, damp_exp: int=2):
        self.stiff_exp = stiff_exp
        self.damp_exp = damp_exp
        self.name = 'exponent_stiff_damp'

    def z_func(self, z):
        dofs = int(z.shape[0]/2)
        return torch.cat((z[:dofs]**self.stiff_exp,z[dofs:]**self.damp_exp), dim=0)
    

