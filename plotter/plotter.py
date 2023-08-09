import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
from matplotlib.legend import _get_legend_handles_labels
import seaborn as sns
import numpy as np
import torch
from typing import Tuple, Union


class Plotter:
    def __init__(self, textwidth: float, fontsize: int, fontname: str, fig_ratio: float, includes_: dict):
        self.textwidth = textwidth  # cm
        self.fontsize = fontsize  # pts
        self.fig_ratio = fig_ratio
        self.includes = includes_

        self.fontpath = os.path.join('plotter', 'fonts', '{}.ttf'.format(fontname))
        if os.path.exists(self.fontpath):
            self.prop = font_manager.FontProperties(fname=self.fontpath)
        else:
            print('Font path not found, reverting to default font.')
            self.fontpath = None
            self.prop = font_manager.FontProperties()
            self.prop.set_weight = 'light'

        self.fig = plt.figure(dpi=150, tight_layout=True)
        self.prop.set_size(self.fontsize)
        self.prop.set_math_fontfamily('cm')
        self.colors = ['tab:blue', 'tab:red', 'tab:gray']
        plt.rc('axes', unicode_minus=False)

    @staticmethod
    def cm_to_inch(value: float) -> float:
        return value / 2.54
    
    @staticmethod
    def sort_data(vec2sort: np.ndarray, *data_: tuple[np.ndarray,...]) -> Union[Tuple[Tuple[np.ndarray,...],np.ndarray], Tuple[np.ndarray,np.ndarray]]:
        sort_ids = np.argsort(vec2sort)
        sorted_data_ = [None] * len(data_)
        for i, data in enumerate(data_):
            sorted_data_[i] = np.zeros_like(data)
            if len(data.shape)>1:
                for j in range(data.shape[1]):
                    sorted_data_[i][:,j] = data[sort_ids,j]
            else:
                sorted_data_[i] = data[sort_ids]
        if len(data_) > 1:
            return tuple(sorted_data_), sort_ids
        else:
            return sorted_data_[0], sort_ids


    def plot_predictions(self, n_dof: int, t_span_obs: np.ndarray, obs_state: np.ndarray, t_span_gt: np.ndarray, predictions: np.ndarray, ground_truth: np.ndarray, task: str) -> plt.Figure:
        """
        Plot the results for predictors.

        Args:
            n_dof (int): Number of degrees of freedom.
            t_span_obs (np.ndarray): time vector for observation data domain
            obs_state (np.ndarray): values of observed state data
            t_span_gt (np.ndarray): time vector for gt/prediction domain
            prediction (np.ndarray): values of predicted state over gt/prediction domain
            ground_truth (np.ndarray): ground truth state values
            task (str): label indicating which task type is being plotted

        Returns:
            plt.Figure: The generated figure.
        """
        spec = gridspec.GridSpec(ncols=n_dof, nrows=2, figure=self.fig)
        spec.update(wspace=0.5, hspace=0.5)  # spacing between subplots

        (ground_truth, predictions, t_span_gt), ids1 = self.sort_data(t_span_gt, ground_truth, predictions, t_span_gt)
        (obs_state, t_span_obs), ids2 = self.sort_data(t_span_obs, obs_state, t_span_obs)

        for kinetic in range(2):
            for dof in range(n_dof):
                f_ax = self.fig.add_subplot(spec[kinetic, dof])

                if kinetic:
                    if n_dof>1:
                        plt.title(r"$\dot{{u}}_{},~\mathrm{ms}^{-1}$".format(dof), fontproperties=self.prop)
                    else:
                        plt.ylabel(r"$\dot{u},~\mathrm{ms}^{-1}$", fontproperties=self.prop)
                else:
                    if n_dof>1:
                        plt.title(r"$u_{},~\mathrm{m}$".format(dof), fontproperties=self.prop)
                    else:
                        plt.ylabel(r"$u,~\mathrm{m}$", fontproperties=self.prop)

                plt.xlabel(r'Time, $s$', fontproperties=self.prop)
                channel = dof + kinetic
                if self.includes["gt"]:
                    plt.plot(t_span_gt, ground_truth[:, channel], label='Exact Solution', color=self.colors[0], linewidth=1.0)
                if self.includes["pred"]:
                    match task:
                        case 'instance':
                            plt.plot(t_span_gt, predictions[:, channel], label='Prediction', linestyle='--', color=self.colors[1], linewidth=1.0)
                        case 'k_plus_1' | 'pgnn':
                            plt.plot(t_span_gt[1:], predictions[:, channel], label='Prediction', linestyle='--', color=self.colors[1], linewidth=1.0)
                if self.includes["obs"]:
                    plt.plot(t_span_obs, obs_state[:, channel], linestyle='None', marker='o', label='Observation Data', color=self.colors[2], markersize=0.15*self.prop.get_size())
                plt.xticks(fontproperties=self.prop)
                plt.yticks(fontproperties=self.prop)

                for axis in ['top','bottom','left','right']:
                    f_ax.spines[axis].set_linewidth(0.65)

        self.fig.legend(*_get_legend_handles_labels([self.fig.axes[0]]), prop=self.prop)
        self.fig.set_figwidth(self.cm_to_inch(self.textwidth))
        self.fig.set_figheight(self.fig_ratio*self.cm_to_inch(self.textwidth))
        self.fig.align_ylabels(self.fig.axes)
        return self.fig

    def show_figure(self):
        plt.show()
