"""CS356 Supplementary material - module of re-used helper functions

Written to accompany lectures for a module called:
CS356 Signal, image, and optical processing
(C) Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created:
tjn, CS, NUIM, 9 XII 2012, quick_show() and quick_correlate_images()
Modified:
tjn, CS, MU, 24 III 2015, quick_plot()
tjn, 8 VI 2015, added normalise parameter to quick_show(), allow arbitrary
    cmap string to quick_show()
tjn, 13 X 2015, converted from Python 2 to Python 3
tjn, 13 XI 2015, added arguments to quick_show()
tjn, 17 XI 2015, allow colorbar thickness adjustment in quick_show()
tjn, 4 XII 2015, modified quick_correlate_images()
tjn, 22 X 2020, updated to work on PythonAnywhere, including FONT_SIZE
tjn, 26 XI 2020, fixed warning raised when subplot was passed non-integers

Tested with Python 3.7.3 on Jupyter Notebook 6.0.3.
"""

from scipy import (array, ones, amax, count_nonzero, sqrt, ceil, isscalar, rint)

# from skimage.exposure import rescale_intensity
from skimage_exposure import rescale_intensity
from skimage.morphology import binary_dilation, disk
from skimage.feature import match_template

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from imageutilssubset import imread_sc

"""

Module-level variables

"""
# Initialise plotting mode
subplot_mode = False
# Initialise subplot grid dimensions to invalid values of the correct type
subplot_rows = 0
subplot_cols = 0
# Initialise subplot index to an invalid value of the correct type
subplot_next_index = 0
# Font size constants that can be overwritten by a caller
FONT_SIZE = None
SUP_FONT_SIZE = None


"""

Functions

"""


def quick_close():
    """A wrapper for modules to close all figures without needing PyPlot as an
    explicit dependency."""
    plt.close('all')


def quick_plot(x,
               f,
               format_str='b-',
               title=None,
               figure_handle=None,
               style=None,
               xlabel=None,
               ylabel=None,
               ylims=None,
               xticklabels=None,
               savefig_suffix=None):
    """Plot a function f over horizontal axis values x.

    Arguments:
    x             : the values on the x-axis on which to plot each point.
    f             : the values to plot.
    format_str    : the plotted line format, passed unmodified to the plot
                    command.
    title         : title on plot.
    figure_handle : a number representing the particular figure to overwrite.
    style         : 'semilogy' for a logarithmically scaled y-axis, or
                    anything else for a regular linearly scaled y-axis plot.
    xlabel        : x-axis label.
    ylabel        : y-axis label.
    ylims         : if a scalar, then this becomes the lower y-axis limit. If
                    a pair then is used for both lower and upper y-axis
                    limits.
    xticklabels   : a list of strings with which to replace the x-axis tick
                    labels.
    savefig_suffix: a suffix to append to the filename (before the dot).
    """
    if figure_handle is not None:
        # Distinguish between None and 0.
        plt.figure(figure_handle)
        # Clear the plot in the existing figure (if any)
        plt.clf()
    else:
        # Create a new figure
        plt.figure()
    if style == 'semilogy':
        plt.semilogy(x, f, format_str)
    elif style == 'semilogx':
        plt.semilogx(x, f, format_str)
    else:
        plt.plot(x, f, format_str)
    plt.axis('tight')
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # If ylims is a scalar/tuple/list then use its values appropriately.
    # Otherwise, ensure that the vertical axis has some decent range so that a
    # signal composed of purely rounding errors is not plotted in 'tight' mode.
    if isscalar(ylims):
        # Apply as lower limit only
        plt.ylim(ylims, plt.ylim()[1])
    elif isinstance(ylims, (list, tuple)) and (len(ylims) == 2):
        # Apply as both lower and upper limits
        plt.ylim(ylims)
    elif (plt.ylim()[1] - plt.ylim()[0]) < 0.5:
        plt.ylim((-1, 1))
    else:
        # Do nothing and let 'tight' mode decide
        pass
    if xticklabels:
        plt.locator_params(axis='x', nbins=len(xticklabels))
        axes = plt.gca()
        # print(axes.get_xticks().tolist())
        axes.set_xticklabels(xticklabels)
    plt.show()
    if savefig_suffix is None:
        savefig_suffix = ''
    plt.savefig(f'graph{savefig_suffix}.png')


def quick_show(im,
               title=None,
               cmap=None,
               colorbar=False,
               subplot=None,
               newsubplotfig=False,
               figure_handle=None,
               fontsize=None,
               suptitle=None,
               supfontsize=None,
               normalise=True,
               axis_off=True,
               tight=False,
               savefig_suffix=None,
               **keywords):
    """Wrapping up boilerplate code to display an image.

    Plots will not be displayed in subplot mode by default.
    If a subplot is required, the subplot grid (height, width) arguments
    should be passed as a 2-tuple through argument "subplot". From then on,
    all plots will be in subplot mode, until the subplot matrix has filled, in
    which case, subplot mode will end.
    If the user wishes to change away from subplot mode before the subplot
    matrix has filled, they need to pass the parameter subplot=False.
    Additional keyword arguments in the dict **keywords are passed blindly
    to plt.imshow().

    Arguments
    ---------
    colorbar : bool or string
        True if a colorbar is to be displayed with the plot or subplot, False
        otherwise.
        If a string, then represents a proportion of the plot to use for the
        colorbar, e.g. '10%' or any other value supported by the 'size'
        parameter to append_axes().
    """
    global subplot_mode, subplot_rows, subplot_cols, subplot_next_index

    def _set_figure(figure_handle, newfig=True):
        """Local function.
        Set the specified figure as active, if appropriate.
        Distinguish between a figure handle of 0 and None.
        Create a new figure, if appropriate.
        """
        if figure_handle is not None:
            plt.figure(figure_handle)
            # Clear the plot in the existing figure (if any)
            plt.clf()
        elif newfig:
            # Create a new figure
            plt.figure()

    def _colorbar_right(size):
        """Local function.
        Create a colorbar to the right of the current axis that is the same
        size as that axis.
        """
        # Create divider for existing axes instance
        divider = make_axes_locatable(plt.gca())
        # Append a new axis to the right of this axis, where 'size'
        # represents a proportion of its width, for example.
        cax = divider.append_axes('right', size=size, pad=0.05)
        # Create colorbar in the appended axes
        plt.colorbar(cax=cax)

    """
    Function begins here.
    """
    # Allow font size to be determined by a caller that injects a value into
    # these module-level constants.
    if fontsize is None:
        fontsize = FONT_SIZE
    if supfontsize is None:
        supfontsize = SUP_FONT_SIZE
    if subplot:
        if newsubplotfig:
            # Create a new figure
            _set_figure(figure_handle)
            # Reset the subplot index counter
            subplot_next_index = 1
        else:
            # Make the appropriate figure active, if specified. Note, the user
            # will have implicitly set newsubplotfig=False.
            # The user will have to be careful here, because the module-level
            # variables controlling the subplot matrix shape, and the next
            # subplot index, are re-used for each plot.
            _set_figure(figure_handle, newfig=False)
            # The subplot layout may have changed in the middle of adding
            # subplots to a figure. Check if layout requires smaller subplots
            # than before, and if so, leave some space (by incrementing the
            # subplot index counter) so that previous subplots are not
            # overwritten.
            if subplot_rows < subplot[0]:
                # Skip the next subplot row
                subplot_next_index += subplot[1]
            elif subplot_cols < subplot[1]:
                # Skip the next subplot column
                subplot_next_index += 1
        # Store these values in global variables so they are persistent
        # between function calls.
        subplot_rows, subplot_cols = subplot
        subplot_mode = True
    elif not subplot_mode:
        # User is not in subplot mode, so create a new figure for each new
        # plot.
        _set_figure(figure_handle)
    elif subplot is False:
        # User was in subplot mode but now wants to change. Note, this False
        # value is intentionally distinct from None.
        subplot_mode = False
        _set_figure(figure_handle)
    elif subplot_next_index > (subplot_rows * subplot_cols):
        # The user was previously in subplot mode, but the subplot matrix is
        # already filled. The user has not specified a new subplot shape so
        # just end subplot mode.
        subplot = False
        subplot_mode = False
        _set_figure(figure_handle)
    else:
        # The user is in normal subplot mode, and has not specified a subplot
        # matrix shape, so re-use the shape from the global variables.
        if newsubplotfig:
            # Create a new figure
            _set_figure(figure_handle)
            # Reset the subplot index counter
            subplot_next_index = 1
        else:
            # Make the appropriate figure active, if specified. Note, the user
            # will have implicitly set newsubplotfig=False.
            # The user will have to be careful here, because the module-level
            # variables controlling the subplot matrix shape, and the next
            # subplot index, are re-used for each plot.
            _set_figure(figure_handle, newfig=False)

    # If a string label is passed rather than a colour map, try to infer the
    # intended colour map. Allow a European spelling for the grey/gray cmap
    # value, and silently adapt to any other misspellings of cmap label.
    if cmap in ('gray', 'grey'):
        cmap = plt.cm.gray
    elif isinstance(cmap, str):
        try:
            cmap = plt.cm.get_cmap(cmap)
        except ValueError:
            cmap = None
    if subplot_mode:
        try:
            plt.subplot(subplot_rows, subplot_cols, subplot_next_index)
        except ValueError as e:
            # Print an error message but continue (potentially populating the
            # remainder of the subplot grid).
            print('Error specifying an index for the subplot. Message is: ' +
                  str(e))
        else:
            subplot_next_index += 1
    if im is None:
        # Make a blank subplot location. Only intended for sub-plot mode.
        pass
        # Hide axis tick labels
        plt.axis('off')
    else:
        # Rescale image to [0, 1] in case it contains negative values
        if normalise:
            im = rescale_intensity(im, out_range=(0, 1))
        # Show the image, passing the cmap, and any additional keyword
        # arguments from dict 'keywords'.
        plt.imshow(im, cmap=cmap, **keywords)
        # Optional titles for figure/subplot
        if title:
            if fontsize:
                plt.title(title, fontsize=fontsize)
            else:
                plt.title(title)
        if suptitle:
            if supfontsize:
                plt.suptitle(suptitle, fontsize=supfontsize)
            elif fontsize:
                supfontsize = rint(int(fontsize) * 1.5)
                plt.suptitle(suptitle, fontsize=supfontsize)
            else:
                plt.suptitle(suptitle)
        # Hide axis tick labels, by default
        if axis_off:
            plt.axis('off')
        # Optional tight layout
        if tight:
            plt.tight_layout()
        # Optional colour bar. This has to go at the bottom, because it
        # creates a new axis object, and subsequent calls to plt.axis('off'),
        # for example, would turn of the ticks on the colour bar rather than
        # on the intended axis.
        if colorbar:
            # If colorbar is a string, then it either refers to a preset
            # colorbar configuration, or else it refers to a string
            # appropriate for the 'size' parameter to append_axes().
            # If colorbar is a float, it refers to the fraction parameter to
            # colorbar().
            if isinstance(colorbar, float):
                # A factor to rescale the colorbar (for example to stop the
                # colorbar overlapping with the title text).
                plt.colorbar(fraction=colorbar)
            elif colorbar == 'small':
                # A factor that stops the colorbar from overlapping with the
                # title text on my particular screen.
                plt.colorbar(fraction=0.04)
            elif colorbar == 'thin':
                # A percentage that looks good on my particular screen, so
                # that suplots are as large as possible.
                _colorbar_right('2%')
            elif isinstance(colorbar, str):
                # Some other percentage string (possibly above percentage is
                # too thin).
                _colorbar_right(colorbar)
            else:
                # If it is simply a bool, or anything else, display the
                # standard colorbar.
                plt.colorbar()
        plt.show()
        if savefig_suffix is None:
            savefig_suffix = ''
        plt.savefig(f'graph{savefig_suffix}.png')


def _highlight_correlations(corr, shape=None, im=None):
    """Display the result of a correlation operation in a pretty way.

    Create an image with the correlation peak positions from 'corr' marked
    with a region of shape 'shape' (if passed) or with some default visible
    shape.
    If 'im' is passed, use it as a faded background, otherwise use a black
    image as background. It is assumed that 'im' is the same size as 'corr'.
    """
    # Create a mask of copies of shape "shape" (if passed) or else calculate
    # a reasonable visible mask. Subtract a couple of pixels to avoid also
    # highlighting closely spaced neighbouring text/objects in the input image.
    if shape:
        mask_element = ones(array(shape) - 3)
    else:
        # The default reasonably visible mask will be circular
        mask_element = disk(max(corr.shape) * 0.02)
    # Find all correlation peaks that have a height within 1% of the height
    # of the maximum correlation peak, and binarise corr based on this
    # threshold.
    corr = (corr > (0.99 * amax(corr)))
    # Count the number of correlation peaks
    num_matches = count_nonzero(corr)
    # Impose the mask element on each correlation peak
    corr = binary_dilation(corr, mask_element)
    # Add the image background (if passed)
    if im is not None:
        # Unhighlight the background image by a factor "unhighlight" keeping
        # the values with the mask elements at 1.0 .
        unhighlight = 0.4
        corr = (corr * (1. - unhighlight)) + unhighlight
        # Ensure "image" is in the range [0, 1] before multiplying
        corr *= rescale_intensity(im, out_range=(0, 1))
    return corr, num_matches


def quick_correlate_images(input_fname, template_fname, method=None):
    """Shell function that calls either Fourier domain correlation or
    normalised cross-correlation.
    """
    # Load the input image, convert to greyscale, convert from uint8 to float,
    # and rescale to the range [0, 1].
    f = imread_sc(input_fname, as_gray=True)
    # Do the same for the template
    t = imread_sc(template_fname, as_gray=True)

    # Perform the chosen method of correlation
    if method:
        ims, titles = method(f, t)
        method_name = method.__name__
    else:
        # By default, apply normalised cross-correlation
        corr = match_template(f, t, pad_input=True)
        # Populate the same variable names used in the other branch of the if
        # statement.
        ims = (corr,)
        titles = ('Correlation output',)
        method_name = match_template.__name__

    # Calculate the arrangement of subplots; ensure subplot is passed ints
    im_count = 3 + len(ims)
    cols = int(ceil(sqrt(im_count)))
    rows = int(ceil(im_count / cols))
    # Make the first two plots
    quick_show(f,
               'Input image',
               cmap='grey',
               subplot=(rows, cols),
               newsubplotfig=True,
               axis_off=False,
               suptitle=('Example using ' + method_name + '()'))
    quick_show(t, 'Template', cmap='grey', axis_off=False)
    # Plot each of the returned (image, title) pairs except the last one
    for im, title in zip(ims[:-1], titles[:-1]):
        quick_show(im, title=title, cmap='grey', axis_off=False)
    # Plot the raw correlation plane
    quick_show(ims[-1], title=titles[-1], colorbar=True, axis_off=False)
    # Plot the cleaned-up correlation plane
    corr, num_matches = _highlight_correlations(ims[-1], shape=t.shape, im=f)
    temp_str = 'Input image with ' + str(num_matches) + ' matches highlighted'
    quick_show(corr, title=temp_str, cmap='grey', axis_off=False)
    # axis tight?
