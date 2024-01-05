"""CS356 Phase of sinusoids - supplementary material

Written to accompany lectures for a module called:
CS356 Signal, image, and optical processing
Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, NUIM, 25 X 2013
Modified:
tjn, CS, MU, 14 XI 2014, converted from GNU Octave to Python
tjn, CS, MU, 22 X 2015, tested with Python 3
tjn, CS, MU, 4 XI 2015, added text to description of complex-valued sinusoids
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

Notes to students:
- See elsewhere for a reminder of some properties of complex values
(definitions: real, imaginary, amplitude, phase, complex sum, multiplying by
a particular phase angle)
- Make sure the file plotsinusoid.py is in your current working directory
- In this worksheet we look in detail at the third basic parameter that
affects a sinusoid (after its amplitude and frequency): its phase angle.
"""

from math import atan2
from numpy import exp, pi, imag, real
import matplotlib
import matplotlib.pyplot as plt

from quickfunctions import quick_close
from plotsinusoid import plot_sinusoid

# matplotlib.rcParams.update({'font.size': 13})
# matplotlib.rcParams.update({'savefig.dpi': 300})
# matplotlib.rcParams['figure.figsize'] = [12, 8]
# matplotlib.rcParams['figure.dpi'] = 150
# %matplotlib notebook
# %matplotlib widget

"""

We've looked at Fourier transforms of pairs of impulses, and the Fourier
transform of a single impulse at the origin, but what is the Fourier transform
of a single off-centre impulse?

"""

# Single off-centre impulses also generate sinusoids.
# Will have non-zero real and imaginary parts in its FT.
quick_close()
plot_sinusoid(2, 0.5, (1., 0.), figs='a')

# This is a single sinusoid, that has both real and imaginary components in
# oscillation -- can you suggest any examples from the real-world of something
# that requires multiple orthogonal descriptions to characterise completely?
# How about the trajectory of a ball in a sports game... will one camera angle
# always be sufficient to record faithfully the path of the ball?

# This sinusoid with equal real/imaginary components has an interesting
# property: its values over time/space trace a perfect circle in the complex
# plane. We can get an intuitive understanding of this by tracing each of the
# real and imaginary components separately in the complex plane.

# This explains why the amplitude and phase values are as they are (constant
# amplitude and linear phase) in the above example.
# [See in-class demo of the values from a complex-valued sinusoid being used to
# trace a circle in the complex plane.]

# Now lets compare this to a single impulse an equal distance on the other
# side of the origin.
plot_sinusoid(2, 0.5, (0., 1.), figs='f')
# You can see that their imaginary values would cancel out, leaving only a
# real-valued sinusoid.
# Equivalently, we can view what happens as the phase values largely cancel
# out, leaving only two possibilities, 0 and pi (0 where the real values are
# positive and pi where the real values are negative).
# The real values do not cancel, and are added together.
# [This concept would scale up to any arbitrary signal/image composed of pairs
# of similar pixels equidistant from the origin (e.g. an image that is
# invariant to reflection through the origin).]

# This is the result of the imaginary components cancelling out:
plot_sinusoid(2, 0.5, (1., 1.), figs='f')

# Here's another example, first the two impulses separately, and then with
# real & imag plots of the impulses combined.
quick_close()
plot_sinusoid(2, 0.5, (1j, 0.), figs='ri')
plot_sinusoid(2, 0.5, (0., -1j), figs='ri')
plot_sinusoid(2, 1, (1j, -1j), figs='ri')

# Here's the same example, first the two impulses separately, and then with
# amplitude & phase plots of the impulses combined.
quick_close()
plot_sinusoid(2, 0.5, (1j, 0.), figs='mp')
plot_sinusoid(2, 0.5, (0., -1j), figs='mp')
plot_sinusoid(2, 1, (1j, -1j), figs='mp')

"""

We saw in one of the first lectures what effect pixel position and pixel
amplitude has on its FT. Now let's understand what effect the phase value of a
pixel has: it changes the phase (shift) of the sinusoid.
(FYI, function 'plot_sinusoid()' was written to reset the amplitude to A=1
each time, independent of impulse value, so only the phase of the impulses
below affects the output.)
We can estimate the following impulse phase values by plotting the impulse
value on the complex plane and estimating visually the phase angle. In the
examples below, the phase angle is calculated for you using the usual arctan
trigonometric rule.

"""


def change_phase_example(d, r_or_i):
    """Example showing change of phase of pixel in Fourier domain.
    """
    for index, a in enumerate(d):
        plot_sinusoid(2, 1, (a, 0.), figs=r_or_i)
        if index == 0:
            tstr = ('Starting phase of sinusoid (showing the ' +
                    str(r_or_i) + ' part only)')
        else:
            tstr = ('Phase shift of ' +
                    str(atan2(imag(a), real(a)) / pi) + 'pi = ' +
                    str(int(atan2(imag(a), real(a)) / (2 * pi) * 360)) + ' deg.')
        plt.title(tstr)


quick_close()
# d = (1., 0.8+0.4j, 0.7+0.7j, 1j, -1.0)
d = (1., 0.7 + 0.7j, 1j, -1.0)
change_phase_example(d, 'r')

"""
In each of the above examples, we plotted the real part only. If we change the
final argument from 'r' to 'i' we will see that there is an identical behaviour
on the imaginary sinusoid
"""
quick_close()
change_phase_example(d, 'i')

"""
Let's repeat the phase shifting behaviour by specifying an explicit phase for
the impulse value (rather than having to figure out what real and imaginary
value is required):
"""
quick_close()
d = (abs(1) * exp(1j * 0),
     # abs(1) * exp(1j * pi / 8),
     abs(1) * exp(1j * pi / 4),
     abs(1) * exp(1j * pi / 2),
     abs(1) * exp(1j * pi))
change_phase_example(d, 'r')
