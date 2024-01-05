"""CS356 Sinusoids and discontinuities - supplementary material

Written to accompany lectures for a module called:
CS356 Signal, image, and optical processing
Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, NUIM, 21 X 2012
Modified:
tjn, CS, MU, 5 XI 2014, converted from GNU Octave to Python
tjn, CS, MU, 13 X 2015, converted from Python 2 to Python 3
tjn, CS, MU, 2 XI 2020, updated for Jupyter Notebook

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
"""

from math import floor, ceil
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from numpy import angle, arange, hstack, imag, ones, real, zeros, linspace, tile
import matplotlib

from quickfunctions import quick_close, quick_plot

matplotlib.rcParams.update({'font.size': 16})
# matplotlib.rcParams.update({'savefig.dpi': 300})
# matplotlib.rcParams['figure.figsize'] = [12, 8]
# matplotlib.rcParams['figure.dpi'] = 150
# %matplotlib notebook
# %matplotlib widget


"""

Module-level names

"""
# Number of elements in array (and in its FT)
M = 1024
# Amplitude of the sinusoid
A = 1
# Horizontal axis values for the plot centred on zero (we could use [0:M-1]
# instead, if we desired).
x = arange(-1 * floor(M / 2.0), ceil(M / 2.0), dtype='int')
# Close all windows left open from previous exercises
quick_close()

"""

Local functions

"""


# Initially, define a function to perform some basic low-pass filtering
def quick_low_pass(a, pixels, zeroval=0, format_str='b-', first='Filtered'):
    """Perform low-pass filtering of 1D array a by removing "pixels" number of
    pixels from each end of the FT of array a.

    Inverse Fourier transform this modified spectrum to see the result.
    Four specific figure handles will be re-used.
    """
    # Four specific figure hanndles
    fh = (10, 20, 30, 40)
    # Fourier transform the input signal
    A = fftshift(fft(a))
    # A non-zero value approximating zero, that allows subsequent signals to be
    # plotted on a log scale.
    zeroval = min(abs(A))
    # Allow a distinction to be made between pixels=0 and pixels is None
    if isinstance(pixels, int):
        # Remove index pixels from one end
        A[:pixels] = zeroval
        # Remove index pixels from the other end (we cannot simply use
        # A[-index:] = zeroval here because it will not produce the expected
        # behaviour when index = 0)
        end = len(A)
        A[end - pixels:] = zeroval
        first += (' (pixels=' + str(pixels) + ')')
    else:
        first += ' (pixels is None)'
        format_str = 'o-'
    # Plot the new Fourier domain in both linear and log plots
    quick_plot(x, abs(A), figure_handle=fh[0], title=first + ' Fourier domain')
    quick_plot(x, abs(A),
               figure_handle=fh[1],
               format_str='o-',
               style='semilogy',
               title=first + ' Fourier domain (log plot)')

    # See what the filtered spectrum looks like back in the signal domain
    afiltered = ifft(ifftshift(A))

    # Plot real and imaginary parts
    quick_plot(x, real(afiltered), format_str, figure_handle=fh[2],
               title=first + ' signal (re values)')
    quick_plot(x, imag(afiltered), format_str, figure_handle=fh[3],
               title=first + ' signal (im values)')
    # Return the filtered signal
    return afiltered


"""

FT of discontinuities (sharp edge and delta function)

"""
# Create a sharp edged step 1D image. [Note, the sum of all the values in
# this image will be one (i.e. it has zero total energy) so we should
# expect to see zero at index 0 of the FT.]
a = hstack((zeros(M // 2) - 1, ones(M // 2)))
quick_plot(x, a, 'o-')
# Plotting its Fourier spectrum shows that this sharp edge is composed
# of a summation of many sinusoids (in principle, a perfect sharp edge
# is composed of an infinite number of sinusoids).
quick_plot(x, abs(fftshift(fft(a))))

# Create an impulse function (approximating a delta function). This is a
# special case in the FT: what is the implication of a non-zero value at
# index 0.
a = hstack((zeros(M // 2), ones(1) * M, zeros((M // 2) - 1)))
quick_plot(x, a, 'o-', 'Original')
# The FT of this real-valued function is a constant real-valued image
quick_plot(x, abs(ifft(ifftshift(a))), 'o-', title='Amplitude of FT')
# The imaginary values are zero. In this special case where the
# imaginary values are zero (but you should be aware that this is never
# true outside of this special case), the phase values are equal to the
# imaginary values and the amplitude values are equal to the real
# values.
quick_plot(x, angle(ifft(ifftshift(a))), 'o-', title='Phase angle of FT')
quick_plot(x, real(ifft(ifftshift(a))), 'o-', title='Real part of FT')
quick_plot(x, imag(ifft(ifftshift(a))), 'o-', title='Imaginary part of FT')

# Demonstrate the inverse of the above:
# The FT of a constant image is an impulse (approximately, a delta
# function).
a = ones(M)
quick_plot(x, a, 'o-', title='Original')
quick_plot(x, abs(fftshift(fft(a))), 'o-', title='Amplitude of FT')

"""

Removing sinusoids from discontinuities
(i) a sharp edge

Note, run these commands one after another in this order. They modify a common
data structure, so start from the first line again if you make any edits.
"""
# Create a step function (different but equivalent to the previous way
# we created one).
quick_close()
a = zeros(M)
a[:M // 2] = 1
quick_plot(x, a, 'o-', title='Original signal')

# Let's look what happens when we perform two FTs on the data, without any
# intermediate filtering.
# As expected, before any low-pass filtering we see that the original FT
# contains sinusoids at all parts of the spatial frequency spectrum.
# An inverse FT without any filtering returns the original signal (performs an
# identity transform).
filtered = quick_low_pass(a, None, first='Original')
# Zoom in to turning point in above plot...
quick_plot(x[479:513], real(filtered[479:513]), title='Zooming in to edge')

# Repeat this, except this time, low-pass filter signal (i.e. crop the Fourier
# spectrum from the edges) before performing the inverse FT. Note, the existing
# plots are modified (scroll up if using a Jupyter norebook).
filtered = quick_low_pass(a, 200)
# Zoom in to turning point in above plot...
quick_plot(x[479:513], real(filtered[479:513]), title='Zooming in to edge')

# Low-pass filter more
filtered = quick_low_pass(a, 400)

# Even more...
filtered = quick_low_pass(a, 500)

# After filtering almost all of the signal,
# all we're left with is a sinusoid...
filtered = quick_low_pass(a, 510)

"""

Removing sinusoids from discontinuities
(ii) a square pulse

"""
quick_close()
# Create a square pulse function
a = zeros(M)
a[300:500] = 1
# The original signal
a = quick_low_pass(a, None, first='Original')

# Low-pass filter several times corresponding to decreasing sample rates
a = quick_low_pass(a, 0, first='Unfiltered')

a = quick_low_pass(a, 200)

a = quick_low_pass(a, 400)

# Notice aliasing
a = quick_low_pass(a, 500)

# At this level of filtering what remains is almost a single sinusoid
a = quick_low_pass(a, 510)

"""

Removing sinusoids from discontinuities
(iii) a sawtooth

"""
quick_close()
# Create a sawtooth function (with 2014 samples)
# f = linspace(0., 1., 100)
f = linspace(0., 1., 128)
f = tile(f, 8)

# Horizontal axis values for the plot centred on zero
M = len(f)
x = arange(-1 * floor(M / 2.0), ceil(M / 2.0), dtype='int')
# View the sawtooth
f = quick_low_pass(f, None, first='Original')

# For reference, return an unfiltered version identical to the input
f = quick_low_pass(f, 0, first='Unfiltered')

# Filter out some sinusoids before the inverse FT (notice aliasing)
f = quick_low_pass(f, 300)

# Filter out some more (notice aliasing, also in imaginary values)
f = quick_low_pass(f, 360)

# Even more (notice aliasing is less evident because there is too much
# blurring).
f = quick_low_pass(f, 380)

# After even more, we just have a basic sinusoid at the same frequency as the
# sawtooth.
# f = quick_low_pass(f, 392)
f = quick_low_pass(f, 501)

# A sawtooth with a different frequency could be created
# f = linspace(0., 1., 256)
# f = tile(f, 4)
