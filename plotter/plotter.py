import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
from matplotlib.legend import _get_legend_handles_labels
import seaborn as sns


class Plotter:
    def __init__(self):

        self.textwidth = 32.0  # cm
        self.fontsize = 10  # pts
        self.fontpath = os.path.join('plotter', 'fonts', 'times.ttf')

        self.fig = plt.figure()
        self.prop = font_manager.FontProperties(fname=self.fontpath)
        self.prop.set_weight = 'light'
        self.prop.set_size(self.fontsize)
        self.colors = sns.color_palette("tab10", 6)
        plt.rc('axes', unicode_minus=False)

    @staticmethod
    def cm_to_inch(value):
        return value / 2.54

    def plot_predictions(self, n_dof, sample, prediction, ground_truth):
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
                plt.plot(sample[:, -1], sample[:, channel], linestyle='None', marker='o', label='Data sample', color=self.colors[0])
                plt.plot(sample[:, -1], prediction[:, channel], label='Predictions', color=self.colors[1])
                plt.plot(ground_truth[:, -1], ground_truth[:, channel], label='Ground truth', color=self.colors[2])
                plt.xticks(fontproperties=self.prop)
                plt.yticks(fontproperties=self.prop)
        self.fig.legend(*_get_legend_handles_labels([self.fig.axes[0]]), prop=self.prop)
        self.fig.set_figwidth(self.cm_to_inch(self.textwidth))
        self.fig.align_ylabels(self.fig.axes)
        return self.fig

    def show_figure(self):
        plt.show()
