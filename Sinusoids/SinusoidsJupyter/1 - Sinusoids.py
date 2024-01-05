"""CS356 Sinusoids - supplementary material

Written to accompany lectures for a module called:
CS356 Signal, image, and optical processing
Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, NUIM, 21 X 2012
Modified:
tjn, CS, MU, 28 X 2014, converted from GNU Octave to Python
tjn, CS, MU, 13 XI 2014, updated interface to quick_plot
tjn, CS, MU, 13 X 2015, converted from Python 2 to Python 3
tjn, CS, MU, 2 XI 2020, updated for Jupyter Notebook

Tested with Python 3.7.3 on Jupyter Notebook 6.0.3.

Implementation note regarding using Spyder locally on your own computer:
My plots in lectures appear in separate windows. You may be happy with them
appearing inline in the IPython console (as they do by default).
If you'd like the plots to appear in separate windows then in the Spyder 2.3.5.2
toolbar go to:
Tools -> Preferences -> IPython Console -> Graphics
and under "Graphics backend" change the "Backend" selection from "Inline" to
"Automatic".

Execution note: This code is not designed to run in its entirety in one go.
Instead, it is designed to be highlighted piece by piece in Spyder and
executed by pressing F9.
Make sure to set the the IPython working directory to this file's
directory so that you can import plotsinusoid.py, for example. This can
be done, by pressing F5 to run the code once in its entirety, 'cd'ing
into this directory in the IPython console, or right clicking on the
tab of this file in Spyder and selecting "Set console working
directory".
"""

from math import ceil
from scipy.fftpack import fft, fftshift
from numpy import arange, sin, zeros, pi
import matplotlib

from quickfunctions import quick_close, quick_plot
from plotsinusoid import add_impulse

matplotlib.rcParams.update({'font.size': 14})
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
x = arange(-1 * M // 2, ceil(M / 2.0), dtype='int')

"""

Local functions

"""


def plot_pair(x, f, F, format_str='b-', title=''):
    """Plot a sinusoid Fourier transform pair."""
    # Plot the space domain
    quick_plot(x, f, format_str, title, xlabel='Space (pixel index)',
               ylabel='Amplitude', savefig_suffix='0')
    # Plot the Fourier domain
    quick_plot(x, F, format_str, 'FT of ' + title[0].lower() + title[1:],
               xlabel='Spatial frequency (pixel index)',
               ylabel='Amplitude', savefig_suffix='1')
    print('Open files graph0.png and graph1.png')


"""

FT of a real-valued sinusoid (looking at absolute values of FT only)

What one should take from this example is that the Fourier transform of a
real-valued sinusoid is a single pair of nonzero pixels equidistant from the
origin. This can be verified by zooming into the plots.

Aside (reflection symmetry):
The Fourier domain has a particular symmetry such that a reflection (through
the origin) of the Fourier domain does not change it, i.e. each pixel has the
same value as its corresponding pixel equidistant from the origin.
We also saw this reflection symmetry in two dimensions with the test image
'sampleshapes.bmp'. This reflection symmetry arises when we Fourier transform
any real-valued signal. We will postpone further discussion of reflection
symmetry until a later worksheet when we Fourier transform complex-valued
signals and images.

"""
# Close all open figures
quick_close()
# Fill a vector of sine values
f = A * sin(x / (12 * pi))
# Just look at the absolute value of the FT
F = abs(fftshift(fft(f)))
# Plot the Fourier transform pair
plot_pair(x, f, F, title='A sinusoid')

"""

Stretch the sinusoid

What one should take from this example is that a lower frequency sinusoid
corresponds to nonzero spatial frequencies that are closer to the origin in the
Fourier domain.

"""
# Fill a vector of sine values
f = A * sin(x / (24 * pi))
# Just look at the absolute value of the FT
F = abs(fftshift(fft(f)))
# Plot the Fourier transform pair
plot_pair(x, f, F, title='Stretched sinusoid (lower frequency)')

"""

Contract the sinusoid

What one should take from this example is that a higher frequency sinusoid
corresponds to nonzero spatial frequencies that are further from the origin in
the Fourier domain.

"""
# Fill a vector of sine values
f = A * sin(x / (2 * pi))
# Just look at the absolute value of the FT
F = abs(fftshift(fft(f)))
# Plot the Fourier transform pair
plot_pair(x, f, F, title='Contracted sinusoid (higher frequency)')

"""

Change the amplitude of the sinusoid

What one should take from this example is that the amplitude of the peak in the
Fourier domain is proportional to the amplitude of the sinudoid in he space
domain.

"""
# Change the amplitude module-level name
A = 0.5 * A
# Fill a vector of sine values
f = A * sin(x / (2 * pi))
# Just look at the absolute value of the FT
F = abs(fftshift(fft(f)))
# Plot the Fourier transform pair
plot_pair(x, f, F, title='Sinusoid with half amplitude')
# Note, this sinusoid contains 26 sinusoids over the length of the vector x[],
# and the amplitude peaks in its Fourier transform are positioned +/- 26 pixels
# from the origin. By zooming in to the image one can verify this. The
# following lines of code also verify it. This relationship between sinusoid
# frequency in the space domain and pixel position in the Fourier domain holds
# for all of the previous examples too (you could copy this code up to the
# previous examples to verify where the peaks are in those examples).
print('The maximum peaks in the FT are at pixel indices:')
# Create a list of all the pixel positions that have value equal to the
# maximum value, and print it out. Convert each pixel position from the
# [0, 2X) range to the [-X, X) range.
print([a - (M // 2) for a in (F >= F.max()).nonzero()[0]])
# A variation of the following might also work:
# np.argpartition(a, (len(a)-1, len(a)-2))[-2:]


"""

Adding and subtracting sinusoids

What one should take from these examples are what the sum of multiple
sinusoids looks like, and how given the sum of a small numbers of sinusoids
(<=3) it is possible to infer what are the frequencies of those sinusoids, and
therefore possible to calculate what the Fourier transform of the sum would
look like.

Some other things to notice are the proportional relationship between the
amplitude of each sinusoid and the amplitude of each the corresponding peaks
in the Fourier domain. Also, addition of sinusoids and subtraction of
sinusoids give rise to a very similar result.

"""
quick_close()
# Reset the amplitude module-level name
A = 1

f = A * sin(x / (32 * pi))
quick_plot(x, f)
F = abs(fftshift(fft(f)))
quick_plot(x, F)

g = 0.3 * A * sin(x / (4 * pi))
quick_plot(x, g)
G = abs(fftshift(fft(g)))
quick_plot(x, G)

h = f + g
quick_plot(x, h)
H = abs(fftshift(fft(h)))
quick_plot(x, H)

k = 0.1 * A * sin(x / (pi / 2))
quick_plot(x, k)
K = abs(fftshift(fft(k)))
quick_plot(x, K)

h = f + g + k
quick_plot(x, h)
H = abs(fftshift(fft(h)))
quick_plot(x, H)

h = f + g - k
quick_plot(x, h)
H = abs(fftshift(fft(h)))
quick_plot(x, H)

"""

Inverse of above: creating sinusoids with impulse functions and the FT

This example reinforces the concept that a Fourier transform is its own
inverse. We can create the same Fourier transform pairs we saw above by
instead starting with the peaks and Fourier transforming them to generate an
appropriate summation of sinusoids.

We generate the peaks by starting with an array of zeros, and carefully
positioning non-zero values symmetrically around the origin (where we choose
the origin to be at index M // 2).
If we set the peak value to half the number of pixels in the array, we will
generate a sinusoid with an amplitude of 1; this is an artifact of the
particular (also, the standard) numerical implementation of the discrete
Fourier transform.

"""

quick_close()
# Create 1D array of zeros
a = zeros(M)

# Add the symmetric nonzero values
a = add_impulse(x, a, 3, M / 2.0)

# Add a second sinusoid with a higher frequency
a = add_impulse(x, a, 13, M / 2.0)

# Add a third sinusoid with an even higher frequency, and smaller amplitude
a = add_impulse(x, a, 30, M / 2.0 * 0.3)

# Another example
quick_close()
a = zeros(M)
a = add_impulse(x, a, 2, M / 2.0)
a = add_impulse(x, a, 5, M / 2.0 * 0.5)
a = add_impulse(x, a, 11, M / 2.0 * 0.3)
