"""CS356 Sinusoids - supplementary material

Written to accompany lectures for a module called:
CS356 Signal, image, and optical processing
Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, NUIM, 21 X 2012
Modified:
tjn, CS, MU, 5 XI 2014, converted from GNU Octave to Python
tjn, CS, MU, 18 XI 2014, added ylims argument to quick_plot
tjn, CS, MU, 19 XI 2014, allowed figs to contain multiple flags
tjn, CS, MU, 22 X 2015, relocated add_impulse() to here
tjn, CS, MU, 11 XI 2015, fix both 0 and 2pi appearing in phase plots due to
    precision/rounding errors
tjn, CS, MU, 22 X 2020, adapt for PythonAnywhere and Jupyter Notebook

Tested with Python 3.7.3 on Jupyter Notebook 6.0.3.

Implementation note: my plots in lectures appear in separate windows because
the projector screen is so small. You may be happy with them appearing inline
in the IPython console (as they do by default). If you'd like the plots to
appear in separate windows then in the Spyder 2.3.1. toolbar go to:
Tools -> Preferences -> IPython Console -> Graphics
and under "Graphics backend" change the "Backend" selection from "Inline" to
"Automatic".
"""

from math import floor, ceil
from numpy import (arange, zeros, sum, real, imag, angle, linspace, pi,
                   isscalar, rint)
from scipy.fftpack import ifft, ifftshift
from scipy import absolute as abs
import matplotlib

from quickfunctions import quick_plot

matplotlib.rcParams.update({'font.size': 16})


def add_impulse(x, a, freq, ampl):
    """Add an impulse pair to the array a, display its FT, and return a.

    The pair of nonzero values will be positioned symmetrically to the
    array, overwriting what was there already.

    Parameters
    ----------

    x : seq
        A list of indices for the plot.

    a : ndarray
        An ndarray (the same shape as x) of data values to be Fourier
        transformed.

    freq : int
        An offset denoting the number of pixels either side of the
        origin to place the pair of impulses. Corresponds to the number
        of periods in the sinusoid.

    ampl : real
        The value of the impulse.

    This function is a stripped-down version of plot_sinusoid() defined
    below that should make it easier to appreciate how simple the
    technique is.
    """
    # The centre index in the array
    origin = len(x) // 2
    a[origin - freq] = ampl
    a[origin + freq] = ampl
    quick_plot(x, a, 'o-', title='Frequencies', savefig_suffix='0')
    A = ifft(ifftshift(a))
    quick_plot(x, real(A), title='Real values', savefig_suffix='1')
    quick_plot(x, imag(A), title='Imaginary values', savefig_suffix='2')
    print('Open files graph0.png, graph1.png, and graph2.png')
    return a


def plot_sinusoid(ocd=1, A=1, d=(-1j, 1j), M=1024, figs='a'):
    """A function to create and plot the sum of arbitrary sinusoids.

    Parameters
    ----------

    ocd : int, or seq of ints, optional, default 1
        Off-centre distance (nonnegative number of pixels that impulse
        functions are positioned on either side of the origin). Also
        corresponds to number of periods in plot. Can be a seq, in
        which case the corresponding impulses are summed prior to
        Fourier transformation, e.g. ocd=(1, 3).

    A : real, or seq of reals, optional, default 1
        Desired amplitude of the sinusoid (assuming a pair of
        impulses). Can have an extra dimension, in which case the
        corresponding impulses are summed prior to Fourier
        transformation, e.g. A=(1, 0.3).

    d : seq of complex, or seq of seq of complex, optional, default(-1j, 1j)
        A pair of unit impulse values in the complex plane that will be
        Fourier transformed to create the sinusoid. Can have an extra
        dimension, in which case the corresponding impulses are summed
        prior to Fourier transformation, e.g. d=((-i i), (-i i)).

    M : int, optional, default 1024
        Length of the array.

    figs : str, optional, default 'a'
        Zero or more of these characters: 'a' all, 'f' final figures,
        'r' final real figure, 'i' final imaginary figure, 'm' final
        amplitude figure, 'p' final phase figure. If `figs` evaluates
        to ``False``, then no figures are displayed.
    """

    def clean_complex(s):
        """Make a succinct str of a str of an array/arrays of complex values.
        """
        subs = ((' 0.-', ' -'),
                (' 0-', ' -'),
                ('-0-', ' -'),
                (' 0.+', ' '),
                (' 0+', ' '),
                ('-0+', ' '),
                ('.+',   '+'),
                ('.-',   '-'),
                (' 1.j', ' i'),
                (' 1j', ' i'),
                ('[1j', '[i'),
                ('(1j', '(i'),
                (' -1.j', '-i'),
                ('-1.j', '-i'),
                (' -1j', '-i'),
                ('-1j', '-i'),
                (' 0.j', ''),
                (' 0j', ''),
                ('+0.j', ''),
                ('+0j', ''),
                ('  ',   ' '),
                ('[ ',   '['),
                (' ]',   ']'),
                ('\n',   ''))
        for (a, b) in subs:
            s = s.replace(a, b)
        return s

    """
    Function starts here
    """
    # If ocd is a scalar, wrap it (and A and d) into tuples of length 1
    if isscalar(ocd):
        ocd = (ocd,)
        A = (A,)
        d = (d,)

    # If user specifies something other than the empty string to indicate that
    # no figures should be displayed, accommodate that.
    if not figs:
        figs = ''

    # Horizontal axis for plotting (centred on zero)
    x = arange(-1 * floor(M / 2.0), ceil(M / 2.0), dtype='int')
    # Centre index of an array with indices 0..M-1
    origin = M // 2
    # Create 1D complex-valued array of zeros
    a = zeros(M, dtype=complex)

    # Add pairs of impulse functions symmetrically about the origin.
    # Set their amplitudes to ensure a particular resulting sinusoid amplitude
    # of A. In the scipy.fftpack implementation of ifft2(), the amplitude of
    # the sinusoid will be sum(abs(a))/len(a) by default, where a is the
    # impulse list.
    for dval, Aval, freq in zip(d, A, ocd):
        ascaling = sum(abs(dval)) / len(a) / Aval
        a[origin - freq] += (dval[0] / ascaling)
        a[origin + freq] += (dval[1] / ascaling)

    # Plot the impulse functions
    if 'a' in figs:
        titlestr = ' values of impulses ' + clean_complex(str(d))
        xstr = 'Spatial frequency (pixel index)'
        quick_plot(x, real(a), 'o-', title='Real'+titlestr, xlabel=xstr)
        quick_plot(x, imag(a), 'o-', title='Imaginary'+titlestr, xlabel=xstr)
        quick_plot(x, abs(a), 'o-', title='Amplitude'+titlestr, xlabel=xstr,
                   ylabel='units of amplitude', ylims=0)
        quick_plot(x, angle(a), 'o-', title='Phase'+titlestr, xlabel=xstr,
                   ylabel='radians', ylims=(-pi, pi))

    # Fourier transform the impulse functions
    A = ifft(ifftshift(a))

    # Create the horizontal axis for plotting in multiples of pi radians
    # (centred on zero) for easy comprehension (set to 1xpi each side of the
    # origin).
    x = linspace(-1, 1, M)
    xstr = 'pi radians'
    titlestr = ' values of FT of impulses ' + clean_complex(str(d))
    if ('a' in figs) or ('f' in figs) or ('r' in figs):
        quick_plot(x, real(A), 'o-', title='Real'+titlestr, xlabel=xstr)
    if ('a' in figs) or ('f' in figs) or ('i' in figs):
        quick_plot(x, imag(A), 'o-', title='Imaginary'+titlestr, xlabel=xstr)
    if ('a' in figs) or ('f' in figs) or ('m' in figs):
        quick_plot(x, abs(A), 'o-', title='Amplitude'+titlestr, xlabel=xstr,
                   ylabel='units of amplitude', ylims=0)
    if ('a' in figs) or ('f' in figs) or ('p' in figs):
        # Fix problems with precision errors causing both 0 and 2pi to appear
        # in the plot, by quantising to small but reasonable number of levels.
        levels = 1000
        two_pi = 2 * pi
        # First convert from the [-pi, pi] range to the [0, 2pi] range so that
        # the % operation will not operate on negative values.
        angleA = (angle(A) + two_pi) % two_pi
        # Then quantise to 'levels' parts in 2pi, and use the modulus to
        # convert all 2pi values to 0.
        q_factor = two_pi / levels
        angleA = (rint(angleA / q_factor) * q_factor) % two_pi
        # Convert from the [-pi, pi] range to the [0, 2pi] range to be more
        # intuitave for beginners. If one wanted to return to remove this line
        # then one should also change the plot argument to ylims=(-pi, pi).
        angleA = (angleA + two_pi) % two_pi
        quick_plot(x, angle(A), 'o-', title='Phase'+titlestr, xlabel=xstr,
                   ylabel='radians', ylims=(-pi, pi))
        quick_plot(x, angleA, 'o-', title='Phase(r)'+titlestr, xlabel=xstr,
                   ylabel='radians', ylims=(0, 2*pi))


# tempstr = 'plotsinusoid.py is a library. It doesn\'t do anything when ' + \
#           'you run it. Instead, use it like ' + \
#          '"from plotsinusoid import plot_sinusoid".'
# print(tempstr)

"""
# Quick test
plt.close('all')
quick_plot(arange(5), arange(5), 'rd-')
quick_plot(arange(5), arange(5), xlabel='x', ylabel='y')
quick_plot(arange(5), linspace(0.0, 0.0001, 5), 'rd-')
from scipy import array
plot_sinusoid(ocd=array([1, 5, 13]),
              A=array([1, 0.3, 0.1]),
              d=array([[1j, -1], [-1j, 1j], [-1, 1]]),
              M=1024,
              figs='a')
plot_sinusoid(array([2]), array([0.5]), array([[1j, 0.]]), figs='r')
plot_sinusoid(array([2]), array([0.5]), array([[1j, 0.]]), figs='i')
plot_sinusoid(array([2]), array([0.5]), array([[0., -1j]]), figs='r')
plot_sinusoid(array([2]), array([0.5]), array([[0., -1j]]), figs='i')
plot_sinusoid(array([2]), array([1]), array([[1j, -1j]]), figs='r')
plot_sinusoid(array([2]), array([1]), array([[1j, -1j]]), figs='i')
"""
