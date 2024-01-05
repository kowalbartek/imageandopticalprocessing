"""CS356 High-, low-, and orientation-band-pass spatial filtering demo

Spatial filtering demonstration. High-, low-, and orientation-band-pass filter
the Fourier spectrum of a two-dimensional greyscale image. Selectively remove
Fourier components of particular orientation.

Written to accommpany lectures for a module called:
CS356 Signal, image, and optical processing
Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, NUIM, 6 IV 2003, created for "Advanced concepts" mini-module
tjn, 23 IX 2011, added round() to constructLine
tjn, 11 X 2012, return value, allow image values to be
    passed directly, write image to disk, show argument
tjn, 11 X 2012, wrote code for enhance_focussed_parts...()
tjn, 17 X 2013 allow greyscale inputs in enhance_...()
tjn, CS, MU, 29 X 2014, converted from GNU Octave to Python
tjn, 29 XI 2014, subplot argument
tjn, 13 X 2015, converted from Python 2 to Python 3
tjn, 29 X 2015, import imread_sc, imsave_sc
tjn, 12 XI 2015, updated arguments to quick_show() after modifications to same
tjn, 6 X 2017, Numpy `ones` will not accept rounded floats any more (only ints)

Tested with Anaconda using Python 3.6.
"""

import os

from numpy import (array, amax, ones, zeros, ceil, dstack,
                   logical_and, logical_or, isscalar)
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy import absolute as abs
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import rotate

from imageutilssubset import window_2d, disc, imread_sc, imsave_sc
from quickfunctions import quick_show


def spatial_filtering_demo(a,
                           demo='freq',
                           param1=0.2,
                           filval=0,
                           show='a',
                           subplot=(2, 3),
                           colorbars=False,
                           fname=None):
    """Spatial filter an image with a hard-edged filter.

    One of two classes of hard-edged spatial frequency filter can be applied
    to the input image: circular (for low-pass and high-pass), and
    orientation-band-pass.

    Argument(s):
    a         : complete path and filename of an image file, or else a 2D
              matrix of appropriate image values (in this case, appropriate
              means greyscale, real-valued, and in the range [0, 1])
    demo      : demo type. 'freq' means that Fourier components are removed or
              retained based on their frequency (default), 'orient' means that
              Fourier components are removed or retained based on their
              orientation.
    param1    : depends on demo type. If demo == 'freq' then param1 is the
              radius of the circular spatial filter, with legal values in the
              range [0, 1] where 0 denotes no aperture and 1 denotes maximum
              allowed aperture size (default is 0.2).
              If demo == 'orient' then param1 is the orientation angle (in
              degrees) of the linear spatial filter (default is 0 degrees).
    filval    : value inside the filter, either 0 (default, denoting high-pass
              filter) or 1 (low-pass filter). Value outside the filter will
              then be 1 or 0, respectively. The strings 'hp' and 'lp' can also
              be passed instead of 0 and 1, respectively.
    show      : 'f' to display the final image only, 'n' to display no images,
              or 'a' to display all images (default).
    subplot   : a pair, denoting the height and width of subplot grid to hold
              all of the displayed images in a single window, or None to
              display each image in its own figure
    colorbars : whether to show colourbars (True) or not (False, default)
    fname     : complete path and filename of image file to write (PNG chosen
              if no file type specified), or '' to write with automatically
              generated filename, or None (default) to not write an output
              file. The parameters of the filtering operation will be added
              automatically to the filename.

    Return value(s):
    a       : (real-valued) filtered image amplitude
    """

    """
    Local functions
    """
    def construct_disc(imshape, diameter=1, filval=1):
        """Return an image of a disc-shaped spatial frequency filter.

        Arguments:
        imshape  : a pair denoting the required dimensions (rows, columns)
        diameter : a normalised proportion of the smaller dimension
        filval   : the value (either 0 or 1) inside the disc
        """
        # Convert the normalised diameter to a number of pixels
        diameter = min(imshape) * diameter

        # Set up the filter values for everywhere OUTSIDE the disc
        f = ones(imshape) - filval

        # Ignore if diameter <= 0; we don't want rounding errors generating
        # any pixels in the disc.
        if diameter > 0:
            # Set only those indices that constitude the disc to filval
            f[disc(diameter, imshape)] = filval
        else:
            # For visualisation purposes only, we set one insignificant pixel
            # to the value of the filter, just to ensure that each filter is
            # bi-valued. If we don't do this, an all-pass filter (a filter
            # containing only 1s) will appear black by default when displayed
            # directly by MathPlotLib.
            f[0, 0] = filval
        return f

    def construct_line(imshape, d, filval, thickness=16):
        """Return an image of a rectangular orientation spatial frequency
        filter.

        Also, complement the filter at its lowest spatial frequencies.

        Arguments:
        imshape   : a pair denoting the shape of the image to be multiplied by
                  the filter.
        d         : is the angle of the orientation filter in degrees.
        filval    : is the value inside the filter aperture, either 1 or 0.
        thickness : is the height of the aperture before rotation (the width of
                  the aperture is dependent on its orientation).
        """

        # Set up the values for everywhere OUTSIDE the filter.
        # Make it as long as the diagonal of the required imshape, using
        # Pythagoras and sqrt(2) = 1.414 .
        f = ones(ceil(array(imshape) * 1.414).astype(int)) - filval

        # Set the values inside the filter (assume a horizontal orientation
        # before rotation).
        # Octave: f[floor(size(f, 1)/2)-8:floor(size(f, 1)/2)+7, :] = filval;
        half_height = f.shape[0] // 2
        # thickness of line to left and right (respectively) of centre
        l_thick = thickness // 2
        r_thick = thickness - l_thick
        f[half_height - l_thick:half_height + r_thick, :] = filval
        # Do it this way in future:
        # aperture = ones((thickness, f.shape[1]))
        # f = window_2d(f, superpose=aperture)

        # Rotate anticlockwise by d degrees
        f = rotate(f, d)

        # Crop filter to required dimensions
        f = window_2d(f, imshape)

        # Remove the lowest spatial frequencies from the filter
        if filval:
            f = logical_and(f, construct_disc(imshape, 0.05, 0))
        else:
            f = logical_or(f, construct_disc(imshape, 0.2, 1))

        return f

    """
    Main body of function starts here
    """

    #
    # Housekeeping
    #
    # Ensure a legal value for show
    if not (isinstance(show, str) and show in 'fna'):
        show = 'a'
    # Ensure a legal value for filval
    if isinstance(filval, str) and filval == 'lp':
        filval = 1
    elif not (isinstance(filval, int) and filval == 1):
        filval = 0
    # Ensure legal values for demo and param1
    if not (isinstance(demo, str) and demo in ('orient', 'freq')):
        demo = 'freq'
    if demo == 'freq':
        if not (isinstance(param1, (int, float)) and (0 <= param1 <= 0.5)):
            param1 = 0.1
    else:
        # demo must be 'orient'
        if not isinstance(param1, (int, float)):
            param1 = 0

    # Read image from file (if necessary) otherwise variable a must already
    # contain appropriate image data.
    if isinstance(a, str):
        # Let Python display the error message to the user if file does not
        # exist.
        a = imread_sc(a)

    # print('This ' + str(a.shape) + ' pixel image now contains ' +
    #      str(a.dtype) + ' values in the range [' + str(amin(a)) + ', ' +
    #      str(amax(a)) + '].')

    if show == 'a':
        quick_show(a,
                   'Original image',
                   cmap='grey',
                   colorbar=colorbars,
                   subplot=subplot,
                   newsubplotfig=True)

    # Fourier transform image
    A = fftshift(fft2(a))
    # Allow full detail of spectrum to be easily appreciated on low dynamic
    # range displays by clipping its values.
    if show == 'a':
        Atemp = abs(A)
        clip_val = amax(Atemp) * 0.001
        quick_show(Atemp.clip(0, clip_val),
                   'Amplitude of Fourier spectrum',
                   cmap='grey')
        # Skip the next subplot location
        quick_show(None)

    # Construct Fourier filter
    if demo == 'freq':
        H = construct_disc(A.shape, param1, filval)
    else:
        # demo must be 'orient'
        H = construct_line(A.shape, param1, filval)
    if show == 'a':
        tempstr = ('Spatial filter (p=' + str(param1) + ', f=' +
                   str(filval)+')')
        quick_show(H, tempstr, cmap='grey', colorbar=colorbars)

    # Filter image
    A *= H
    if show == 'a':
        tempstr = ('Spectrum (p=' + str(param1) + ', f=' +
                   str(filval) + ')')
        # Re-use the clip_val from the pre-filtered spectrum
        quick_show((abs(A)).clip(0, clip_val),
                   tempstr,
                   cmap='grey')

    # Inverse Fourier transform and display filtered image.
    # Ensure filtered image is real and appropriately scaled.
    a = rescale_intensity(abs(ifft2(ifftshift(A))))
    if show != 'n':
        tempstr = ('Image amplitude, (p=' + str(param1) +
                   ', f=' + str(filval) + ')')
        quick_show(a, tempstr, cmap='grey')

    # print('This ' + str(a.shape) + ' pixel image now contains ' +
    #      str(a.dtype) + ' values in the range [' + str(amin(a)) + ', ' +
    #      str(amax(a)) + '].')

    # Write filtered image to disk (if required).
    # If no explicit write file name is specified, auto-generate one.
    # Set output file format to PNG if no filename extension present.
    if fname != None:
        # Write filtered colour image to disk
        root, ext = os.path.splitext(fname)
        if root == '':
            root = 'untitled'
        if ext == '':
            ext = '.png'
        fname = root + '[p=' + str(param1) + ',f=' + str(filval) + ']' + ext
        imsave_sc(fname, a)
        print('File "' + fname + '" written to disk.')

    return a


def enhance_focussed_parts_of_colour_image(fname, filter_radii):
    """Function to enhance high-spatial-frequency (in general, focussed) parts
    of an RGB image and supress all low-spatial-frequency (in general, out of
    focus) parts. The filter radii can be a single scalar (a single radius)
    and can be a list/tuple of radii, in which case the enhanced images (one
    for each radius) are summed.

    No explicit input argument checking is performed.

    Example usage:
    enhance_focussed_parts_of_colour_image('green.jpg', 0.05)
    enhance_focussed_parts_of_colour_image('green.jpg', (0.05, 0.09, 0.1))
    """

    # Ensure filter_radii is a list
    if isscalar(filter_radii):
        filter_radii = [filter_radii]
    else:
        filter_radii = list(filter_radii)

    # Read the colour image file, convert uint8 values to floats, and rescale
    # to the range [0, 1]
    a = imread_sc(fname, as_gray=False)

    # Make the image RGB if it has the dimensions of a greyscale image
    if a.ndim == 2:
        a = dstack([a] * 3)

    # Plot in its own figure (specify that subplot mode has finished)
    quick_show(a, 'Original colour image', cmap='grey', subplot=False)

    # Initialise the filtered greyscale image
    filt_greyimage = zeros((a.shape[0], a.shape[1]))
    for r in filter_radii:
        # Spatially high-pass filter a greyscale version of the input image.
        # Output filtered image will be real valued and will have values
        # automatically rescaled to the range [0, 1].
        # img_as_float() is not needed because we are sure that image a does
        # not contain values of type uint8.
        temp = spatial_filtering_demo(rgb2gray(a), 'freq', r, 'hp', 'a')
        filt_greyimage += temp

    # Rescale to [0,1]
    filt_greyimage = rescale_intensity(filt_greyimage)

    # Image a is a N*M*3 (colour) image and we wish to multiply each RGB colour
    # channel by the 2D array filt_greyimage.
    a[:, :, 0] = a[:, :, 0] * filt_greyimage  # red
    a[:, :, 1] = a[:, :, 1] * filt_greyimage  # green
    a[:, :, 2] = a[:, :, 2] * filt_greyimage  # blue

    # Create a string representation of the numeric parameters used in this
    # computation (without any spaces).
    filter_radius_str = str(filter_radii).replace(' ', '')

    # Display filtered colour image in own figure
    quick_show(a,
               'Filtered colour image (filter ' + filter_radius_str + ')',
               subplot=False)
    # Write filtered colour image to disk
    root, ext = os.path.splitext(fname)
    fname = root + filter_radius_str + 'infocus' + '.png'
    imsave_sc(fname, a)
    print('File "' + fname + '" written to disk.')
