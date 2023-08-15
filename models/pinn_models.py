import torch
import torch.nn as nn
import models.pinn_systems as systems
import models.pinn_nonlinearities as nonlinearities
from typing import Tuple

class sdof_pinn(nn.Module):
    '''
    Arbitrary SDOF PINN
    '''

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seq_len: int, config: dict):

        super().__init__()
        self.n_input = input_dim
        self.n_output = output_dim
        self.n_hidden = hidden_dim
        self.seq_len = seq_len
        self.n_layers = 4  # arbitrarily set for now
        self.activation = nn.Tanh

        self.build_net()

        self.configure(config)

    def build_net(self) -> int:
        '''
        Builds the MLP
        '''
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_input * self.seq_len, self.n_hidden * self.seq_len),
            self.activation(),
            nn.Sequential(*[nn.Sequential(*[
                nn.Linear(self.n_hidden * self.seq_len, self.n_hidden * self.seq_len),
                self.activation()
                ]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden * self.seq_len, self.n_output * self.seq_len),
            nn.Unflatten(dim=1, unflattened_size=(self.seq_len, self.n_output))
            )
        return 0
    
    def forward(self, t: torch.Tensor, G: torch.Tensor=torch.tensor([0.0]), D: torch.Tensor=torch.tensor([1.0])) -> torch.Tensor:
        '''
        Forward pass of the network.

        Args:
            t: vector of time samples
            G: Dirichlet boundary conditions values
            D: Dirichlet boundary conditions mask
        '''
        x = self.net(t)
        y = G + D * x
        return y
    
    def configure(self, config) -> int:
        '''
        Configures the PINN model
        '''
        self.config = config

        print('Setting\instantiating physical parameters in PINN....')
        self.system_discovery = config['system_discovery']
        match config['phys_system_type']:
            case 'duffing_sdof':
                self.system = systems.sdof_pinn_system(
                    m_ = config['m_'],
                    c_ = config['c_'],
                    k_ = config['k_'],
                    kn_ = config['kn_'],
                    nonlinearity = nonlinearities.exponent_stiffness(exponent=3)
                    )
                self.config['nonlinearity'] = 'exponent_damping'
            #TODO: Add more cases of system here
        if self.system_discovery:
            self.m_ = config['m_'].requires_grad_()
            self.register_parameter('c_', nn.Parameter(torch.ones(1,1)))
            self.register_parameter('k_', nn.Parameter(torch.ones(1,1)))
            self.register_parameter('kn_', nn.Parameter(torch.ones(1,1)))

        print('Setting normalisation constants....')
        self.alpha_t = torch.tensor(config['alphas'][2]).float()
        self.alpha_z = torch.tensor(config['alphas'][:2]).reshape(-1,1).float()
        self.alpha_f = torch.tensor(config['alphas'][3]).float()
        for param_name, norm in config["param_norms"].items():
            setattr(self,"alpha_"+param_name,torch.tensor(norm).float())

        return 0
    
    def set_switches(self, lambdas: dict) -> int:
        '''
        Switches are used to speed up pinn if scaling parameters are set to zero
        '''
        print('Setting loss function switches....')
        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        self.switches = switches
        return 0
    
    def loss_func(self, lambdas: dict, t_obs: torch.Tensor, z_obs: torch.Tensor, t_col: torch.Tensor, f_col: torch.Tensor) -> Tuple[float, list, dict]:
        '''
        Calculates the individual losses

        Args:
            t_obs: vector of time inputs in observation domain
            z_obs: matrix of state observations in observation domain
            t_col: vector of time inputs in collocation domain
            f_col: vector of force values in collocation domain

        Returns:
            loss: total loss as sum of individuals
            losses: list of values of individual losses multiplied by their scaling parameter
            residuals: dictionary of residual vectors/matrices for each loss

        '''

        t_obs = t_obs.requires_grad_()
        z_obs = z_obs.requires_grad_()
        t_col = t_col.requires_grad_()
        f_col = f_col.requires_grad_()

        if self.switches['obs']:
            # generate prediction in observation domain
            zp_obs_hat = self.forward(t_obs)
            R_obs = zp_obs_hat.reshape(-1,2) - z_obs.reshape(-1,2)
        else:
            R_obs = torch.zeros((2,2))

        if self.switches['ode'] or self.switches['cc']:
            # generate prediction in collocation domain
            zp_col_hat_ = self.forward(t_col)

            # retrieve derivatives
            dxdt_ = torch.zeros_like(zp_col_hat_)
            for i in range(zp_col_hat_.shape[2]):
                dxdt_[:,:,i] = torch.autograd.grad(zp_col_hat_[:,:,i], t_col, torch.ones_like(zp_col_hat_[:,:,i]), create_graph=True)[0]  # ∂_t-hat N_z(i)-hat

            # reshape for loss functions
            t_col_flat, col_sort_ids = torch.sort(t_col.reshape(-1))
            zp_col_hat = zp_col_hat_.reshape(-1,2)[col_sort_ids,:]
            dxdt = dxdt_.reshape(-1,2)[col_sort_ids,:]
            f_col = f_col.reshape(-1,1)[col_sort_ids,:]

        if self.switches['ode']:

            # retrieve physical parameters
            if self.system_discovery:
                m_ = self.m_
                c_ = self.c_ * self.alpha_c
                k_ = self.k_ * self.alpha_k
                if self.system.nonlinearity.name == 'exponent_stiffness':
                    kn_ = self.kn_ * self.alpha_kn
                else:
                    kn_ = torch.zeros((1,1))
                self.system.update_modal_matrices(m_, c_, k_, kn_)
            A, H, An = self.system.state_matrices()

            match self.config['nonlinearity']:
                case None:
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zp_col_hat.T) - H@(self.alpha_f*f_col.T)
                    R_ode = R_[1,:]
                case _:
                    zn = self.system.nonlin_state_transform(self.alpha_z*zp_col_hat.T)
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zp_col_hat.T) - An@(zn) - H@(self.alpha_f*f_col.T)
                    R_ode = R_[1,:]
        else:
            R_ode = torch.zeros((2))

        if self.switches['cc']:
            # continuity condition residual
            R_cc = R_[0,:]
        else:
            R_cc = torch.zeros((2))

        if self.switches['ic']:
            ic_id = torch.argwhere(t_col_flat==torch.tensor(0.0))
            if ic_id.nelement()==0:
                R_ic = torch.zeros(2)
            else:
                R_ic = zp_col_hat[ic_id,:].T
        else:
            R_ic = torch.zeros(2)

        residuals = {
            "R_obs" : R_obs,
            "R_cc" : R_cc,
            "R_ode" : R_ode,
            "R_ic" : R_ic
        }

        L_obs = lambdas['obs'] * torch.mean(torch.sum(R_obs**2, dim=1), dim=0)
        L_cc = lambdas['cc'] * torch.mean(R_cc**2, dim=0)
        L_ode = lambdas['ode'] * torch.mean(R_ode**2, dim=0)
        L_ic = lambdas['ic'] * torch.mean(R_ic**2,dim=0)

        loss = L_obs + L_cc + L_ode +L_ic
        return loss, [L_obs, L_cc, L_ode, L_ic], residuals

    
