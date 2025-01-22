import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from chainconsumer import ChainConsumer
import src.spherical_harmonics as SH

RS = SH.RealSphericalHarmonics()

def compare_estimate_to_reality(a_reality, a_estimate, ylm_mat=None, nside=None,
                                mat_P=None):
    """
    Plot various figures comparing the resonstructed map to the original map,
    and the residuals of the estimated alm.

    Parameters
    ----------
    a_reality, a_estimate
        alm vectors corresponding to the fiducial map and the estimated map
        respectively. a_reality should be as long or longer than a_estimate.
    ylm_mat
        Ylm matrix corresponding to a_reality. This is chopped down to use for 
        a_estimate too. If none is passed, one will be created.
    nside
        If a Ylm matrix is being created, must pass a value for its nside. If 
        a Ylm matrix is being passed, the nside argument is ignored.
    mat_P
        If the observation matrix is passed, plots the pointing directions in
        the sky reconstruction plot.
    """
    try:
        lmax = RS.get_lmax(len(a_reality))
        lmod = RS.get_lmax(len(a_estimate))
        if lmod > lmax:
            msg = 'fiducial alm vector must be as long or longer than the ' \
                + 'estimated alm vector.'
            raise ValueError(msg)
    except ValueError:
        raise ValueError('invalid alm vector length.')
    
    if ylm_mat is None:
        if nside is None:
            raise ValueError("if Ylm matrix isn't passed, nside should be.")
        ylm_mat = SH.calc_spherical_harmonic_matrix(nside=nside, lmax=lmax)
    nside = hp.npix2nside(len(ylm_mat[:,0]))
    
    no_modes = len(a_estimate)
    ylm_mat_mod = ylm_mat[:, :no_modes]

    hp.mollview(ylm_mat@a_reality, title=f"lmax={lmax} original map")
    plt.show()

    hp.mollview(ylm_mat_mod@a_estimate, title=f"lmod={lmod} reconstructed map")
    plt.show()

    hp.mollview(ylm_mat_mod@a_estimate - ylm_mat@a_reality, 
                title=f"lmax={lmax}, lmod={lmod} reconstructed map residuals")
    plt.show()

    hp.mollview(np.abs(ylm_mat_mod@a_estimate - ylm_mat@a_reality), 
                title=f"lmax={lmax}, lmod={lmod} reconstructed map abs(residuals)")
    
    if mat_P is not None:
        # Use np.argmax to find the column index of the first occurrence of 1 in each row.
        indxs = np.argmax(mat_P, axis=1).tolist()
        thetas, phis = hp.pix2ang(nside=nside, ipix=indxs)
        hp.projscatter(thetas, phis, alpha=0.4, s=3, color='r')
    plt.show()

    mode_residuals = abs(a_estimate[:no_modes] - a_reality[:no_modes])
    mode_residuals /= abs(a_reality[:no_modes])
    lmod_arr = list(range(0, lmod+1, 2))
    lmod_idx = [RS.get_idx(l=l, m=-l) for l in lmod_arr]
    plt.axhline(y=1, color='k', alpha=0.7)
    plt.semilogy(range(no_modes), mode_residuals, '.')
    plt.xticks(ticks=lmod_idx, labels=lmod_arr)
    plt.xlabel("spherical harmonic l number")
    plt.ylabel("|alm estimated - alm fiducial|/|alm fiducial|")
    plt.show()
    

def compare_reconstructions(a_reality, *a_estimates, labels=None, fmts=None, ylm_mat=None,
                            return_comparisons_fig=False):
    """
    Compare the errors in reconstructions of the original alm vector by plotting
    the residuals of the vectors and optionally a bar chart of the different temperature
    standard deviations from the fiducial map.

    Parameters
    ----------
    a_reality, a_estimates
        Fiducial alm vector and set of estimated alm vectors. Estimated vectors
        must be as long or shorter than the fiducial vector.
    labels
        List of labels for the different estimates.
    fmts
        List of scatter plot format arguments for the estimates.
    ylm_mat
        ylm matrix corresponding to a_reality. If passed, calculates the
        residuals of the reconstructed temperature maps for each of the
        estimates and produces a bar plot.
    return_comparisons_fig : bool
        If True, will return the figure comparing the alm to one another and to
        the fiducial alm.
    """
    no_modes_list = [len(a) for a in a_estimates]
    max_modes = np.max(no_modes_list)
    lmod_max = RS.get_lmax(max_modes)
    lmod_max_arr = list(range(0, lmod_max+1, 2))
    lmod_max_idx = [RS.get_idx(l=l, m=-l) for l in lmod_max_arr]

    if labels is None:
        labels = ['']*len(a_estimates)
    if fmts is None:
        fmts = ['.']*len(a_estimates)
    
    fig, ax = plt.subplots(1, 1)
    for a_estimate, label, fmt, no_modes in zip(a_estimates, labels, fmts, no_modes_list):
        mode_residuals = abs(a_estimate[:no_modes] - a_reality[:no_modes])
        mode_residuals /= abs(a_reality[:no_modes])
        ax.semilogy(range(no_modes), mode_residuals, fmt, label=label)
    ax.axhline(y=1, color='k', alpha=0.7)
    ax.legend()
    ax.set_xticks(ticks=lmod_max_idx, labels=lmod_max_arr)
    ax.set_xlabel("spherical harmonic l number")
    ax.set_ylabel("|alm estimated - alm fiducial|/|alm fiducial|")
    ax.set_ylim(1e-7, 1e+5)
    if return_comparisons_fig:
        return fig
    fig.show()
    plt.close('all')

    # Make a bar chart comparing the reconstruction error in rms temperature.
    if ylm_mat is not None:
        map_reality = ylm_mat @ a_reality
        ylm_mats_trunc = [ylm_mat[:,:len(a)] for a in a_estimates]
        map_estimates = [y_mat @ a for y_mat, a in zip(ylm_mats_trunc, a_estimates)]
        rms = [np.std(map_reality - map_est) for map_est in map_estimates]
        plt.bar(x=list(range(len(rms))), height=rms)
        plt.xticks(ticks=list(range(len(rms))), labels=labels)
        plt.ylabel("RMS temperature error [K]")
        plt.xticks(rotation=30)
        plt.show()
    
    if ylm_mat is not None:
        lmax = RS.get_lmax(len(a_reality))
        idxs = [RS.get_idx(l=l, m=l)+1 for l in range(lmax)]
        cutoff_realities = np.array([ylm_mat[:,:i] @ a_reality[:i] for i in idxs])
        for a, label, fmt in zip(a_estimates, labels, fmts):
            lmax_a = RS.get_lmax(len(a))
            idxs = [RS.get_idx(l=l, m=l)+1 for l in range(lmax_a)]
            cutoff_estimates = np.array([ylm_mat[:,:i] @ a[:i] for i in idxs])
            rms = np.std(cutoff_estimates-cutoff_realities[:len(idxs)], axis=1)
            plt.semilogy(list(range(lmax_a)), rms, fmt, label=label)
        plt.xlabel("spherical harmonic l number")
        plt.ylabel("RMS temperature residuals [K]")
        plt.legend()
        plt.show()

class AxesCornerPlot():
    """
    Allows easy axis limit and label customisation on top of ChainConsumer
    corner plots. Each parameter has both a tag (a simple string 
    representation) and a label (to use when plotting). If no labels are
    passed, will use the tags as the labels.
    """
    def __init__(self, *tagged_chains : dict, labels=None, param_truths=None, plotter_kwargs={}):
        """
        Allows easy axis limit and label customisation on top of ChainConsumer
        corner plots. Each parameter has both a tag (a simple string 
        representation) and a label (to use when plotting). If no labels are
        passed, will use the tags as the labels.

        Parameters
        ----------
        tagged_chains : dict
            Dictionaries containing the flattened chains with parameter tags as
            keys, and chains as values. Optionally also contains a 'config' key,
            whose value is a dictionary of keyword arguments to be passed to
            ChainConsumer.add_chain(). This can be used e.g. to set the chain
            names.
        labels : list of str, optional
            List containing the labels to plot the corner plot with. Distinct
            from tags to allow easy references to otherwise long, LaTeX markup 
            string labels. If not passed, uses the chain's tags as labels.
        param_truths : list of floats, optional
            List containing the fiducial parameter values.
        plotter_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the ChainConsumer.plotter
            method.
        """
        # Check labels and param_truths lengths are the same as the dimension of 
        # all the tagged chains.
        if labels is not None:
            for tagged_chain in tagged_chains:
                chain_dim = len([key for key in tagged_chain.keys() \
                                 if key!='config'])
                if len(labels) != chain_dim:
                    errmsg = "Number of tagged chain parameters for each chain " \
                           + "should equal number of labels, but lengths are " \
                           + f"{chain_dim} and {len(labels)}"
                    raise ValueError(errmsg)
        else:
            try:
                labels = list(tagged_chains[0].keys())
            except IndexError:
                raise TypeError("No chains were passed.")
        
        if param_truths is not None:
            if len(labels) != len(param_truths):
                errmsg = "Number of tagged chain parameters should equal " \
                       + "number of parameter truths, but lengths are " \
                       + f"{len(tagged_chain)} and {len(param_truths)}"
                raise ValueError(errmsg)

        self.dim    = len(labels)
        self.labels = labels
        self.tags   = tagged_chains[0].keys()

        self.consumer = ChainConsumer()
        for tagged_chain in tagged_chains:
            try:
                chain_kwargs = tagged_chain.pop('config')
            except KeyError:
                chain_kwargs = {}
            self.consumer.add_chain(tagged_chain, **chain_kwargs)
        
        self.cornerfig = self.consumer.plotter.plot(truth=param_truths, **plotter_kwargs)
        self.axiscube  = np.reshape(self.cornerfig.axes, (self.dim, self.dim))

        # Set the labels of the corner plot, overriding the default
        # ChainConsumer behavior of setting the labels to the keys of the chain
        # dictionary passed.
        for tag, label in zip(self.tags, self.labels):
            self.set_label(tag, label)
            try:
                self.set_xticks(tag, ticks='plain')
            except ValueError:
                pass
            try:
                self.set_yticks(tag, ticks='plain')
            except ValueError:
                pass
    
    def _get_row_axes(self, tag):
        """
        Return the row of axes of the parameter corresponding to 'tag' in order
        from left to right.
        """
        tag_index = list(self.tags).index(tag)
        if tag_index == 0:
            raise ValueError(f"element '{tag}' has no row axes.")
        
        return self.axiscube[tag_index, :tag_index]
    
    def _get_column_axes(self, tag):
        """
        Return the column of axes of the parameter corresponding to 'tag' in 
        order from top to bottom.
        """
        tag_index = list(self.tags).index(tag)
        if tag_index == (self.dim-1):
            raise ValueError(f"element '{tag}' has no column axes.")
        
        return self.axiscube[tag_index-(self.dim-1):,tag_index]

    def _get_hist_axes(self, tag):
        """
        Return the histogram of axes of the parameter corresponding to 'tag'.
        """
        tag_index = list(self.tags).index(tag)
        return self.axiscube[tag_index,tag_index]

    def set_tick_params(self, **kwargs):
        """
        (Re)set the tick parameters in the corner plot, e.g. labelsize.
        Keyword arguments are passed to matplotlib.axes.Axes.tick_params. 
        See the documentation for more information.
        """
        x_axes = self.axiscube[-1]
        y_axes = self.axiscube[1:,0]
        for axes in x_axes:
            axes.tick_params(axis='x', **kwargs)
        for axes in y_axes:
            axes.tick_params(axis='y', **kwargs)

    def set_labelpad(self, labelpad):
        """
        Set the padding between the axes and the axes labels.
        """
        x_axes = self.axiscube[-1]
        y_axes = self.axiscube[1:,0]
        tick_label_heights = []
        for axes in x_axes:
            axes.xaxis.set_label_coords(0.5, -labelpad)
        for axes in y_axes:
            axes.yaxis.set_label_coords(-labelpad, 0.5)

    def set_figurepad(self, figpad):
        """
        Set the padding between the left and bottom sides of the figure if the
        labels are being cut off.
        """
        self.cornerfig.subplots_adjust(bottom=figpad, left=figpad)

    def set_yticks(self, tag, ticks, ticklabels=None, **kwargs):
        """
        Set the yticks and ticklabels of the parameter corresponding to 'tag'.
        Any kwargs provided are passed to matplotlib.axes.Axes.set_yticks.
        See the documentation of this method for more information.

        If ticks='plain' is passed, calls 
            Axes.ticklabel_format(style='plain', useOffset=False)
        to remove scientific notation.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        try:
            ylabel_axes = self._get_row_axes(tag)[0]
        except ValueError:
            raise ValueError(f"'{tag}' has no y-axis.")
        if ticks == 'plain':
            ylabel_axes.ticklabel_format(axis='y', style='plain', useOffset=False)
        else:
            ylabel_axes.set_yticks(ticks, ticklabels, **kwargs)

    def set_xticks(self, tag, ticks, ticklabels=None, **kwargs):
        """
        Set the xticks and ticklabels of the parameter corresponding to 'tag'.
        Any kwargs provided are passed to matplotlib.axes.Axes.set_xticks.
        See the documentation of this method for more information.
        
        If ticks='plain' is passed, calls 
            Axes.ticklabel_format(style='plain', useOffset=False)
        to remove scientific notation.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        
        try:
            xlabel_axes = self._get_column_axes(tag)[-1]
        except ValueError:
            # The last tag in the tag list is a histogram axes - special case.
            if tag == list(self.tags)[-1]:
                xlabel_axes = self._get_hist_axes(tag)
        if ticks == 'plain':
            xlabel_axes.ticklabel_format(axis='x', style='plain', useOffset=False)
        else:
            xlabel_axes.set_xticks(ticks, ticklabels, **kwargs)
    
    def set_label_sizes(self, labelsize):
        """
        (Re)set the font size of the labels in the corner plot.
        """
        for tag, label in zip(self.tags, self.labels):
            self.set_label(tag, label, fontsize=labelsize)
    
    def set_label(self, tag, label, **kwargs):
        """
        (Re)set the plotting label of the parameter corresponding to 'tag'. Any
        keyword arguments are passed to the set_xlabel and set_ylabel methods 
        of matplotlib.axes.Axes.
        See the documentation of either of these methods for more information.

        Returns dict with keys 'xlabel' and 'ylabel' with values of the label
        successfully set. If a label was not set successfully (e.g. if setting
        the ylabel of the first parameter in the corner plot), the value will be
        None.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        
        label_dict = {'xlabel':None, 'ylabel':None}
        try:
            ylabel_axes = self._get_row_axes(tag)[0]
            ylabel_axes.set_ylabel(label, **kwargs)
            label_dict['xlabel'] = label
        except ValueError:
            pass
        try:
            xlabel_axes = self._get_column_axes(tag)[-1]
            xlabel_axes.set_xlabel(label, **kwargs)
            label_dict['ylabel'] = label
        except ValueError:
            # The last tag in the tag list is a histogram axes - special case.
            if tag == list(self.tags)[-1]:
                ylabel_axes = self._get_hist_axes(tag)
                ylabel_axes.set_xlabel(label, **kwargs)
                label_dict['xlabel'] = label
                
        # Update the label attribute with the updated label if the re-labelling
        # was successful.
        if label_dict['xlabel'] is not None or label_dict['ylabel'] is not None:
            tag_index = list(self.tags).index(tag)
            self.labels[tag_index] = label
        return label_dict

    def set_lim(self, tag, lower=None, upper=None, **kwargs):
        """
        Set the plotting limits of the parameter corresponding to 'tag'. Any
        keyword arguments are passed to the set_xlim and set_ylim methods of
        matplotlib.axes.Axes.
        See the documentation of either of these methods for more information.

        Returns dict with keys 'xlim' and 'ylim' with values of the limits
        successfully set. If a limit was not set successfully (e.g. if setting
        the ylim of the first parameter in the corner plot), the value will be
        None.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        
        lim_dict = {'xlim':None, 'ylim':None}
        try:
            row_axes = self._get_row_axes(tag)
            for axes in row_axes:
                axes.set_ylim(bottom=lower, top=upper, **kwargs)
            lim_dict['ylim'] = (lower, upper)
        except ValueError:
            pass
        try:
            column_axes = self._get_column_axes(tag)
            for axes in column_axes:
                axes.set_xlim(left=lower, right=upper, **kwargs)
            lim_dict['xlim'] = (lower, upper)
        except ValueError:
            pass
        
        # All parameters have a hist axis.
        hist_axes = self._get_hist_axes(tag)
        hist_axes.set_xlim(left=lower, right=upper, **kwargs)

    def get_figure(self):
        """
        Return the corner plot.
        """
        return self.cornerfig
