import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Plotter:
    def __int__(self):
        print('hello')

        # TODO need to put all constants for standardized plotting

    def plot_predictions(self, n_dof, sample, prediction, ground_truth):
        fig = plt.figure(constrained_layout=True)
        spec = gridspec.GridSpec(ncols=n_dof, nrows=2, figure=fig)
        for j in range(2):
            for i in range(n_dof):
                f_ax = fig.add_subplot(spec[j, i])
                channel = 2 * i + j
                plt.plot(sample[:, -1], sample[:, channel], 'o')
                plt.plot(sample[:, -1], prediction[:, channel])
                plt.plot(ground_truth[:, -1], ground_truth[:, channel])
        return fig

    def show_figure(self):
        plt.show()
