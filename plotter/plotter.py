import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
from matplotlib.legend import _get_legend_handles_labels
import seaborn as sns
import numpy as np


class Plotter:
    def __init__(self, textwidth: float, fontsize: int, fontname: str):
        self.textwidth = textwidth  # cm
        self.fontsize = fontsize  # pts

        self.fontpath = os.path.join('plotter', 'fonts', '{}.ttf'.format(fontname))
        if os.path.exists(self.fontpath):
            self.prop = font_manager.FontProperties(fname=self.fontpath)
        else:
            print('Font path not found, reverting to default font.')
            self.fontpath = None
            self.prop = font_manager.FontProperties()

        self.fig = plt.figure()
        self.prop.set_weight = 'light'
        self.prop.set_size(self.fontsize)
        self.colors = sns.color_palette("tab10", 6)
        plt.rc('axes', unicode_minus=False)

    @staticmethod
    def cm_to_inch(value: float) -> float:
        return value / 2.54

    def plot_predictions(self, n_dof: int, sample: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray, t_span: np.ndarray, ground_t_span) -> plt.Figure:
        """
        Plot the predictions for the given data.

        Args:
            n_dof (int): Number of degrees of freedom.
            sample (np.ndarray): Input data sample.
            prediction (np.ndarray): Model predictions.
            ground_truth (np.ndarray): Ground truth data.
            t_span (np.ndarray): Time vector.

        Returns:
            plt.Figure: The generated figure.
        """
        spec = gridspec.GridSpec(ncols=n_dof, nrows=2, figure=self.fig)
        spec.update(wspace=0.5, hspace=0.5)  # spacing between subplots

        for kinetic in range(2):
            for dof in range(n_dof):
                f_ax = self.fig.add_subplot(spec[kinetic, dof])

                if kinetic:
                    plt.title(r"$\dot{{x}}_{}(t)$".format(dof), fontproperties=self.prop)
                    plt.ylabel(r"$\frac{m}{s}$", fontproperties=self.prop)
                else:
                    plt.title(r"$x_{}(t)$".format(dof), fontproperties=self.prop)
                    plt.ylabel(r"m", fontproperties=self.prop)

                plt.xlabel('s', fontproperties=self.prop)
                channel = 2 * dof + kinetic
                plt.plot(t_span, sample[:, channel], linestyle='None', marker='o', label='Data sample', color=self.colors[0])
                plt.plot(t_span[1:], prediction[:, channel], label='Predictions', color=self.colors[1])
                plt.plot(ground_t_span, ground_truth[:, channel], label='Ground truth', color=self.colors[2])
                plt.xlim([ground_t_span.min(), ground_t_span.max()])
                plt.xticks(fontproperties=self.prop)
                plt.yticks(fontproperties=self.prop)

        self.fig.legend(*_get_legend_handles_labels([self.fig.axes[0]]), prop=self.prop)
        self.fig.set_figwidth(self.cm_to_inch(self.textwidth))
        self.fig.align_ylabels(self.fig.axes)
        return self.fig

    def show_figure(self):
        plt.show()
