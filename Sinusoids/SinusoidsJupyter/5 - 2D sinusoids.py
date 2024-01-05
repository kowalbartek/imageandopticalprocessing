"""CS356 2D sinusoids - supplementary material

Written to accompany lectures for a module called:
CS356 Image and optical processing
(C) Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, NUIM, 1 XII 2013
Modified:
tjn, CS, MU, 21 XI 2014, converted from GNU Octave to Python
tjn, 22 X 2015, tested with Python 3
tjn, 12 XI 2015, uses quickfunctions and imageutilssubset modules
tjn, 16 XI 2015, modified single_pixel() and added all_pixels()
tjn, 11 XI 2016, modified single_pixel() examples
tjn, CS, MU, 12 XI 2020, updated for Jupyter Notebook

Tested with Python 3.7.3 on Jupyter Notebook 6.0.3.

Implementation note: my plots in lectures appear in separate windows because
the projector screen is so small. You may be happy with them appearing inline
in the IPython console (as they do by default). If you'd like the plots to
appear in separate windows then in the Spyder 2.3.1. toolbar go to:
Tools -> Preferences -> IPython Console -> Graphics
and under "Graphics backend" change the "Backend" selection from "Inline" to
"Automatic".

Execution note: This code is not designed to run in its entirety in one go.
Instead, it is designed to be highlighted piece by piece in Spyder and
executed by pressing F9.

In this worksheet we look at 2D sinusoids and their FTs.
"""

import os
from numpy import (mgrid, pi, sin, real, imag, zeros, rot90, amax, ceil,
                   random, angle)
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from numpy import absolute as abs
import matplotlib

from quickfunctions import quick_close, quick_show
from imageutilssubset import imread_sc, imsave_sc, create_animated_gif

# matplotlib.rcParams.update({'font.size': 13})
# matplotlib.rcParams.update({'savefig.dpi': 300})
# matplotlib.rcParams['figure.figsize'] = [12, 8]
# matplotlib.rcParams['figure.dpi'] = 150
# %matplotlib notebook
# %matplotlib widget

"""

Consider the 2D function: f(x,y) = A sin(ux + vy).
We'll create a local function to plot this mathematical function, and plot its
Fourier transform.

"""


def image_sinusoid(u, v, M=512, N=512, A=1):
    """Display a 2D sinusoid with horizontal and vertical spatial frequency
    parameters u and v, and display its FT.

    Arguments:
    u, v : vertical and horizontal spatial frequency parameter, respectively
    M, N : dimensions of the image
    A    : amplitude of the sinusoid
    """
    # Create coordinate arrays for vectorised evaluations of functions
    R, C = mgrid[0:M, 0:N]
    # Create 2D sinusoid that is composed of a sinusoid horizontally and
    # a sinusoid vertically.
    f = A * sin(u * R + v * C)
    # print('The image has shape ' + str(f.shape) + ' pixels.')
    # Show a figure with two subplots
    quick_show(f,
               title='Sinusoid',
               cmap='grey',
               subplot=(1, 2),
               newsubplotfig=True,
               normalise=False,
               fontsize=20)
    F = abs(fftshift(fft2(f)))
    quick_show(F,
               title='Amplitude of FT of sinusoid',
               cmap='grey',
               normalise=False,
               axis_off=False,
               tight=True,
               fontsize=20)


"""
This first image is a diagonal sinusoid because there are both u and v
components (i.e. it has spatial frequency both in horizontal and in vertical
directions).
"""
quick_close()
# Vertical spatial frequency parameter
u = 1. / (8. * pi)
# Horizontal spatial frequency parameter
v = 1. / (8. * pi)
# Display the sinusoid and its FT
image_sinusoid(u, v)

"""
Sinusoid varying in horizontal direction only (no vertical spatial frequency)
"""
# Vertical spatial frequency parameter
u = 0.
# Horizontal spatial frequency parameter
v = 1. / (8. * pi)
# Display the sinusoid and its FT
image_sinusoid(u, v)

"""
Sinusoid varying in vertical direction only (no horizontal spatial frequency)
"""
# Vertical spatial frequency parameter
u = 1. / (8. * pi)
# Horizontal spatial frequency parameter
v = 0
# Display the sinusoid and its FT
image_sinusoid(u, v)

"""
Sinusoid varying by different amounts vertically and horizontally
"""
# Vertical spatial frequency parameter
u = 1. / (4. * pi)
# Horizontal spatial frequency parameter
v = 1. / (8. * pi)
# Display the sinusoid and its FT
image_sinusoid(u, v)

"""
Very high spatial frequencies in the vertical direction; when u < pi we
get well-sampled spatial frequencies but with some minor aliasing as u
approaches pi; when u >= pi we get highly-corruptive aliasing.
u = 0.5*pi  results in approximately four samples per sinusoid
u = 0.75*pi  results in less than three samples per sinusoid
u = 0.99*pi  results in approximately two samples per sinusoid
u = pi  results in a highly-corrupted aliased image
"""
quick_close()
# Vertical spatial frequency parameter
u = 0.99 * pi  # 2 samples per sinusoid
# Horizontal spatial frequency parameter
v = 0
# Display the sinusoid and its FT
image_sinusoid(u, v)

"""
Approaching zero (approaching a constant image). This is a lowest spatial
frequency possible: a single period.
"""
# Vertical spatial frequency parameter
u = 0
# Horizontal spatial frequency parameter
u = 1. / (32. * pi)
# Display the sinusoid and its FT
image_sinusoid(u, v)

"""

Creating 2D sinusoids using FT

"""


def image_impulses(d, M=1023, N=1023, A=1.):
    """Display a 2D image with two impulses positioned symmetrically about the
    origin, and display its FT.

    Parameters
    ----------

    d : seq of pairs
        Each pair denotes a vertical and horizontal distance,
        respectively, (in pixels) of an impulse from the origin. Each
        specified impulse implies a second impulse located symmetrically
        about the origin.

    M, N : ints
        Dimensions of the image.

    A : complex
        The value of the implulses.

    I've made the default image dimensions odd (1023x1023) so that the rot90
    'trick' shown below will work conveniently. Try yourself with a 1024x1024
    image to see that you'll get off-centre impulse pairs. An even dimension
    image (e.g. 1024x1024) will admit a much more efficient FT calculation, and
    if one wishes that, then the impulses will have to be positioned
    asymetrically, such as a[513, 512] = 1; a[513, 514] = 1.
    """
    # Create a blank image into which the impulses will be inserted
    a = zeros((M, N), dtype=complex)
    # The array indices of the origin of the blank image
    originY = M // 2
    originX = N // 2
    # Insert each specified impulse
    for h, v in d:
        a[originY + h, originX + v] = A
    # For each impulse, insert a second, symmetrically positioned, impulse
    a += rot90(a, 2)

    # Show a figure with two subplots, and display the impulses
    quick_show(abs(a),
               title='Amplitude of impulses',
               cmap='grey',
               subplot=(1, 2),
               newsubplotfig=True,
               normalise=False,
               axis_off=False,
               fontsize=20)
    # Fourier transform
    A = ifft2(ifftshift(a))
    quick_show(real(A),
               title='Real of FT of impulses',
               cmap='grey',
               normalise=False,
               tight=True,
               fontsize=20,
               colorbar=True)


quick_close()

# Horizontal
image_impulses([(0, 1)])

# Horizontal, higher frequency
image_impulses([(0, 21)])

# Vertical
image_impulses([(5, 0)])

# Diagonal
image_impulses([(5, 5)])

# Multiple sinusoids
image_impulses([(0, 2), (2, 0)])

# Multiple sinusoids
image_impulses([(3, 5), (0, 2)])

# Multiple sinusoids in same direction
image_impulses([(0, 1), (0, 22)])
image_impulses([(0, 3), (0, 22)])

"""

Evidence that any signal can be decomposed into sinusoids.

Here we take a single pixel from the FT of a familiar image and perform an
inverse FT on it. We know that the inverse FT of the whole Fourier spectrum is
the original image, so we can understand that all of the real and imaginary
sinusoids must combine to make that original image.

"""

quick_close()
fname = 'sampleshapes'
fext = '.bmp'

# Show the original space-domain and Fourier-domain images
a = imread_sc(fname + fext)
quick_show(a, cmap='grey', title='Original image', axis_off=False)
Fa = fftshift(fft2(a))
quick_show(abs(Fa),
           cmap='grey',
           title='Fourier amplitude (from which we take individual pixels )',
           normalise=False,
           vmax=0.001 * amax(abs(Fa)))


def single_pixel(Fa, r, c, showfigs=True):
    """Study the space-domain of a single Fourier domain pixel.

    Fa is the Fourier transform of a well-known image.
    r and c are either scalars representing a single pair of coordinates, or
    two lists/tuples representing pairs of coordinates.
    The corresponding space-domain is examined, plotted if appropriate, and
    returned.
    """
    mask = zeros(Fa.shape, dtype=float)
    mask[r, c] = 1.
    # Calculate the corresponding space domain
    s = ifft2(ifftshift(Fa * mask))
    if showfigs:
        if isinstance(r, int):
            title_str = 'Amplitude pixel at ' + str((r, c))
        elif len(r) <= 3:
            title_str = 'Amplitude pixels at ' + str((r, c))
        else:
            title_str = 'Amplitude pixels (' + str(len(r)) + ' of)'
        quick_show(abs(Fa * mask),
                   cmap='grey',
                   subplot=(1, 3),
                   newsubplotfig=True,
                   title=title_str,
                   axis_off=False)
        quick_show(real(s),
                   cmap='grey',
                   title='Real part of inverse FT')
        quick_show(imag(s),
                   cmap='grey',
                   title='Imag part of inverse FT')
    # Return the space domain
    return s


# A pixel far from the origin. In this example, and in the others below, we
# see we have both real and imaginary terms in the output. This is because, as
# has been shown in previous demos, a single pixel in the space domain
# specifies a complex-valued sinusoid in the Fourier domain.
s = single_pixel(Fa, 220, 200)

# A pixel closer to the origin
s = single_pixel(Fa, 100, 125)

# A pixel very close to the origin
s = single_pixel(Fa, 127, 125)

# Multiple pixels together
s = single_pixel(Fa,
                 (120, 134, 100, 102, 130, 150),
                 (150, 134, 100, 102, 110, 150))


def all_pixels(fpath, num_frames=4, view='ampl', showfigs=False):
    """Study the space-domain of increasing numbers of Fourier domain pixels.

    A demonstration that iteratively accumulates complex-valued sinusoids,
    generated by Fourier transforming one (or more) complex-valued pixels at a
    time from the FT of an input image. In this demo, the real part (and
    the amplitude part) of the accumulator gradually takes the shape of the
    input image, and the imaginary part (and the phase part) takes on lower
    and lower values until it consists only of rounding errors.

    The result is an animated GIF written to disk, and (optionally) the
    display of a figure with subplots for frames.

    Paramaters
    ----------

    fpath is the path to an image file that can be opened using
    skimage.io.imread(). It is opened as a greyscale image.

    num_frames is the desired number of frames in the video (first frame
    containing one sinusoid, last frame equal to the input), so the minumum
    number of frames expected is 2.

    view is the view of the complex-valued accumulator that should be
    displayed as a real-valued image, either 'ampl'/'abs' (default),
    'phase'/'phas'/'angle', 'real', or 'imag'.

    showfigs denotes whether a subplot should be displayed showing
    simultaenously each frame in the animation. An upper limit on the number
    of subplots is hardcoded here.
    """

    def get_temp_fnames(num_pixels, view):
        """Generate a list of filenames that will be used to store the
        individual frames for the video. Also, generate the animation
        filename.
        Argument num_pixels is a list of numbers of Fourier-domain pixels
        added to each frame, respectively.
        """
        # Function-level constant
        TEMP_DIR = 'temp_files'
        # Give the filename a default file extension if it does not have one
        fpath_root, ext = os.path.splitext(fpath)
        # Ensure that the temporary directory exists
        dir_name, fname_root = os.path.split(fpath_root)
        temp_dir = os.path.join(dir_name, TEMP_DIR)
        try:
            os.makedirs(temp_dir)
            print('Creating directory "' + temp_dir +
                  '" as it does not exist.')
        except FileExistsError:
            # Ignore an exception caused by the directory existing already
            pass
        # Create a list of temporary filenames to store each decoded frame
        # (has to be a list rather than a generator because it will be
        # accessed twice).
        temp_fname = os.path.join(temp_dir, fname_root)
        temp_fnames = [temp_fname + '_' + str(n) + '.png' for n in num_pixels]
        return temp_fnames, (fpath_root + '_anim_' + view)

    # Set the default view if incorrect argument
    if view not in ('ampl', 'abs', 'phase', 'phas', 'angle', 'real', 'imag'):
        view = 'ampl'
    # Read the input image from disk and get its FT
    a = imread_sc(fpath)
    Fa = fftshift(fft2(a))
    # Cross product of the set of row indices and set of column indices
    indices = [(r, c) for r in range(Fa.shape[0]) for c in range(Fa.shape[1])]
    # Randomise the order
    random.shuffle(indices)
    # Unzip the list of index pairs into two separate lists, one for row
    # indices and one for column indices. This allows a convenient way to
    # index multiple pixels in an ndarray. zip() produces tuples, so convert
    # to lists.
    row, col = zip(*indices)
    row, col = list(row), list(col)
    # Create complex-valued accumulator array
    acc = zeros(Fa.shape, dtype=complex)
    # Calculate the number of new Fourier-domain pixels to add in each frame
    # (rounded up).
    step = int(ceil(len(row) / (num_frames - 1)))
    # Calculate the number of Fourier-domain pixels added cumulatively in each
    # frame.
    num_pixels = list(range(0, len(row), step)) + [len(row)]
    # Generate a list of filenames to temporarily store each frame of the
    # animation.
    temp_fnames, anim_fname = get_temp_fnames(num_pixels, view)
    # Generate the Fourier domain masks corresponding to the new pixels added
    # for each frame of the animation.
    for fname, p, q in zip(temp_fnames, [0] + num_pixels, num_pixels):
        # Choose a number of pixel coordinates to Fourier transform (from the
        # end of the randonly-generated list), and return as an image of
        # space-domain complex-valued sinusoids.
        # Add the complex-valued sinusoids to the accumulator
        acc += single_pixel(Fa, row[p:q], col[p:q], showfigs=False)
        # Write the frame to disk
        if view in ('phase', 'phas', 'angle'):
            acc_view = angle(acc)
        elif view is 'real':
            acc_view = real(acc)
        elif view is 'imag':
            acc_view = imag(acc)
        else:
            acc_view = abs(acc)
        imsave_sc(fname, acc_view)
        if showfigs:
            raise NotImplementedError('Showing figures not implemented yet.')

    # Create an animated GIF from these decoded frames
    create_animated_gif(temp_fnames, delay=200, out_fname=anim_fname)


quick_close()
all_pixels('sampleshapes.bmp', view='real', num_frames=100)
all_pixels('sampleshapes.bmp', view='imag', num_frames=100)
all_pixels('sampleshapes.bmp', view='ampl', num_frames=100)
all_pixels('sampleshapes.bmp', view='phas', num_frames=100)

"""
Try on the command line in Linux (allows pausing):
    mplayer -loop 0  -speed 0.5 sampleshapes_anim_imag.gif
or in a web browser, or in any video player that supports animated GIFs.
"""
