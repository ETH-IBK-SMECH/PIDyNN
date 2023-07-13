import torch
from typing import Tuple

class sdof_pinn_system():
    '''
    SDOF PINN system class
    '''

    def __init__(self, m_, c_, k_, kn_, nonlinearity=None):

        self.nonlinearity = nonlinearity
        self.A = torch.zeros((2,2)).requires_grad_()
        self.H = torch.zeros((2,1)).requires_grad_()
        self.An = torch.zeros((2,1)).requires_grad_()
        self.update_modal_matrices(m_, c_, k_, kn_)
        if nonlinearity is not None:
            self.nonlin_state_transform = lambda z: nonlinearity.z_func(z).requires_grad_()

    def state_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.A, self.H, self.An

    def update_modal_matrices(self, m_: torch.Tensor, c_:torch.Tensor, k_: torch.Tensor, kn_: torch.Tensor) -> int:

        # self.A = torch.tensor([[0, 1],[-k_/m_, -c_/m_]]).requires_grad_()
        self.A = torch.cat((
            torch.cat((torch.zeros((1,1)), torch.ones((1,1))), dim=1),
            torch.cat((-torch.linalg.inv(m_)@k_, -torch.linalg.inv(m_)@c_), dim=1)
        ), dim=0).requires_grad_()
        # self.H = torch.tensor([0,1/m_]).reshape(-1,1).requires_grad_()
        self.H = torch.cat((torch.zeros((1,1)), torch.linalg.inv(m_)), dim=0).requires_grad_()
        if self.nonlinearity is not None:
            match self.nonlinearity.name:
                case 'exponent_stiffness':
                    # self.An = torch.tensor([0,-kn_/m_]).reshape(-1,1).requires_grad_()
                    self.An = torch.cat((torch.zeros((1,1)), -torch.linalg.inv(m_)@kn_), dim=0).requires_grad_()
                    # self.An[1,0] = -kn_/m_
        return 0

class mdof_pinn_system():
    '''
    Base class for mdof pinn system
    '''

    def __init__(self, dofs: int):
        self.dofs = dofs

    def state_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.A, self.H, self.An

class cantilever(mdof_pinn_system):

    def __init__(self, m_, c_, k_, cn_, kn_, dofs=None, nonlinearity=None):

        if type(m_) is torch.Tensor:
            dofs = m_.shape[0]
        elif dofs is not None:
            m_ = m_ * torch.ones((dofs))
            c_ = c_ * torch.ones((dofs))
            k_ = k_ * torch.ones((dofs))
        else:
            raise Exception('Under defined system, please provide either parameter vectors or number of degrees of freedom')

        self.nonlinearity = nonlinearity
        self.update_modal_matrices(m_, c_, k_, cn_, kn_)

        if nonlinearity is not None:
            self.nonlin_state_transform = lambda z : nonlinearity.z_func(torch.cat((
                z[:dofs] - torch.cat((torch.zeros(1),z[:dofs-1])),
                z[dofs:] - torch.cat((torch.zeros(1),z[dofs:-1]))
            ), dim=0))

        super().__init__(dofs)

    def update_modal_matrices(self, m_: torch.Tensor, c_: torch.Tensor, k_: torch.Tensor, cn_: torch.Tensor, kn_: torch.Tensor) -> int:

        self.dofs = m_.shape[0]
        self.M = torch.diag(m_) 
        self.C = torch.diag(torch.cat((c_[:-1]+c_[1:],c_[-1:]),axis=0)) + torch.diag(-c_[1:],diagonal=1) + torch.diag(-c_[1:],diagonal=-1)
        self.K = torch.diag(torch.cat((k_[:-1]+k_[1:],k_[-1:]),axis=0)) + torch.diag(-k_[1:],diagonal=1) + torch.diag(-k_[1:],diagonal=-1)

        self.A = torch.cat((
            torch.cat((torch.zeros((self.dofs,self.dofs)), torch.eye(self.dofs)), dim=1),
            torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
        ), dim=0)

        self.H = torch.cat((
            torch.zeros((self.dofs, self.dofs)), torch.linalg.inv(self.M)
        ), dim=0)

        if self.nonlinearity is not None:
            match self.nonlinearity.name:
                case 'exponent_damping':
                    self.Cn = torch.diag(cn_) - torch.diag(cn_[1:], diagonal=1)
                    self.An = torch.cat((torch.zeros((self.dofs,self.dofs)), -torch.linalg.inv(self.M)@self.Cn), dim=0)
                case 'exponent_stiffness':
                    self.Kn = torch.diag(kn_) - torch.diag(kn_[1:], diagonal=1)
                    self.An = torch.cat((torch.zeros((self.dofs,self.dofs)), -torch.linalg.inv(self.M)@self.Kn), dim=0)
                case 'exponent_stiff_damp':
                    self.Cn = torch.diag(cn_) - torch.diag(cn_[1:], diagonal=1)
                    self.Kn = torch.diag(kn_) - torch.diag(kn_[1:], diagonal=1)
                    self.An = torch.cat((
                        torch.cat((torch.zeros((self.dofs,self.dofs)), torch.eye(self.dofs)), dim=1),
                        torch.cat((-torch.linalg.inv(self.M)@self.Kn, -torch.linalg.inv(self.M)@self.Cn), dim=1)
                    ), dim=0)
        return 0
    
