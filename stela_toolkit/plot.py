import matplotlib.pyplot as plt


class Plotter:
    """
    Flexible wrapper around matplotlib for plotting binned or unbinned spectral results.
    Handles default formatting, error bars, labels, and saving.
    """
    
    @staticmethod
    def plot(x=None, y=None, xerr=None, yerr=None, **kwargs):
        """
        Generalized plotting method for spectrum-like data.

        Parameters:
        ----------
        x : array-like
            x-axis values (e.g., frequencies).
        
        y : array-like
            y-axis values (e.g., power or cross-power values).
        
        xerr : array-like, optional
            Uncertainties in the x-axis values (e.g., frequency widths).
        
        yerr : array-like, optional
            Uncertainties in the y-axis values (e.g., power uncertainties).
        
        **kwargs: Additional keyword arguments for customization.
        """

        if xerr is not None:
            xerr = xerr if len(list(xerr)) > 0 else None
        if yerr is not None:
            yerr = yerr if len(list(yerr)) > 0 else None

        if x is None or y is None:
            raise ValueError("Both 'x' and 'y' must be provided.")

        title = kwargs.get('title', None)

        # Default plotting settings
        if yerr is not None or xerr is not None:
            default_plot_kwargs = {'color': 'black', 'fmt': 'o', 'ms': 3, 'lw': 1.5, 'label': None}
        else:
            default_plot_kwargs = {'color': 'black', 's': 3, 'label': None}

        figsize = kwargs.get('figsize', (8, 4.5))
        fig_kwargs = {'figsize': figsize, **kwargs.pop('fig_kwargs', {})}
        plot_kwargs = {**default_plot_kwargs, **kwargs.pop('plot_kwargs', {})}
        major_tick_kwargs = {'which': 'major', **kwargs.pop('major_tick_kwargs', {})}
        minor_tick_kwargs = {'which': 'minor', **kwargs.pop('minor_tick_kwargs', {})}
        savefig_kwargs = kwargs.pop('savefig_kwargs', {})
        save = kwargs.pop('save', None)

        plt.figure(**fig_kwargs)

        if yerr is not None:
            if xerr is not None:
                plt.errorbar(x, y, xerr=xerr,yerr=yerr, **plot_kwargs)
            else:
                plt.errorbar(x, y, yerr=yerr, **plot_kwargs)
        else:
            if xerr is not None:
                plt.errorbar(x, y, xerr=xerr, **plot_kwargs)
            else:
                plt.scatter(x, y, **plot_kwargs)

        # Set labels if provided
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)

        if xlabel:
            plt.xlabel(xlabel, fontsize=12)
        if ylabel:
            plt.ylabel(ylabel, fontsize=12)

        plt.xscale(kwargs.get('xscale', 'linear'))
        plt.yscale(kwargs.get('yscale', 'linear'))

        # Show legend if label is provided
        if plot_kwargs.get('label'):
            plt.legend()

        if title:
            plt.title(title)

        # Tick kwargs
        major_tick_kwargs.setdefault('which', 'both')
        major_tick_kwargs.setdefault('direction', 'in')
        major_tick_kwargs.setdefault('length', 6)
        major_tick_kwargs.setdefault('width', 1)
        major_tick_kwargs.setdefault('labelsize', 12)
        major_tick_kwargs.setdefault('top', True)
        major_tick_kwargs.setdefault('right', True)

        plt.tick_params(**major_tick_kwargs)
        if len(minor_tick_kwargs) > 1:
            plt.minorticks_on()
            plt.tick_params(**minor_tick_kwargs)

        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        if save:
            plt.savefig(save, **savefig_kwargs)

        plt.show()
