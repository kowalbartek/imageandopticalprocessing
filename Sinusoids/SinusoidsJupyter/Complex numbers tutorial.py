"""CS356 Complex numbers tutorial

Written to accompany lectures for a module called:
CS356 Signal, image, and optical processing
Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, MU, 11 XI 2014
Last modified: tjn, CS, MU, 13 X 2015, converted from Python 2 to Python 3

Tested with Python 3.4.3 shipped with Anaconda 2.3.0 (64-bit) on Ubuntu 14.04.

Execution note: This code is not designed to run in its entirety in one go.
Instead, it is designed to be highlighted piece by piece in Spyder and executed
by pressing F9.
"""

from numpy import exp, pi, real, imag, angle, sin, cos, conj

# An arbitrary complex value
a = 3 + 4j
print(a)

# Real part
print(real(a))

# Imaginary part
print(imag(a))

"""
This form of representing complex numbers is called Cartesian form. It gets
its name from the straightforward way that complex numbers can be visualised
when the real and imaginary parts are treated as cartesian coordinates.
The following code plots arbitrary complex values in the complex plane.
"""
# This was done on the whiteboard instead of in software


"""
When manipulating complex numbers, it is sometimes more mathematically
elegant to choose a different but equivalent coordinate system: polar
coordinates. In this coordinate system, each point in the complex plane is
uniquely identified by two properties: an amplitude and a phase.
"""
# Amplitude
print(abs(a))

# Phase (or phase angle)
print(angle(a))

# The phase angle is given in radians by default
phase = angle(a)
tempstr = 'A phase angle of ' + str(phase) + ' radians is equivalent to ' + \
          str(phase / (2 * pi) * 360.0) + ' degrees.'
print(tempstr)

"""
A phase-only value (or to give it its full name, a phase-only complex value) is
a complex number with an amplitude of 1.

Randomly generate phase-only values and plot them. This should create a unit
circle in the complex plane.
"""
# This was drawn on the whiteboard instead


"""
exp() is an exponential function using Euler's number e
Euler's relationship is e^ib = cos(b) + i sin(b)
"""
# Euler's relationship can be verified for any arbitrary complex value
# using the code below, for which the Python function exp() can be used
# to raise a number to the power of Euler's number e.
b = 5.0 + 11.0j  # An arbitrary complex value
print(exp(1j * b))
print(cos(b) + 1j * sin(b))
# These two expression print the same value, with minor precision errors

# Recall, to combine a real value a and an imaginary value b into a single
# complex number we use the relationship a+ib. It's slightly more complicated
# for amplitudes and phases. To combine an amplitude A and a phase p into a
# single complex number, we use the relationship Ae^ip, which with Python is
# A * exp(1j * p).
a = 3 + 4j
amplitude = abs(a)
phase = angle(a)
print('A complex number with amplitude {} and phase {} is represented '
      'internally by Python as {}.'.format(amplitude,
                                           phase,
                                           amplitude * exp(1j * phase)))
# We can see from the output of the print statement above that the
# internal representation of complex numbers in Python is Cartesian
# form, even when the programmer uses polar form.

# An amplitude of 1 and an angle of pi/2 (90 deg.) represents the complex
# number i.
print(1 * exp(1j * pi / 2))
# An amplitude of 1 and an angle of pi (180 deg.) represents the complex
# number -1. There is an implicit amplitude of 1 in the calculation below when
# we specify the phase angle but do not specify an amplitude.
print(exp(1j * pi))


"""
Adding a complex number, adding an angular phase value

1. You knew already that to add a complex number C to complex number D, we must
separately add the real and imaginary components of C and D.

2. To add a phase value (or to give it its full name, an angular phase value) p
to a complex number C, it is a little more complicated. We have to multiply C by
a phase-only number with phase p. Recall from the diagram of the complex plane
drawn in class, adding a phase value to a complex number is equivalent to a
counter-clockwise rotation around the origin.
"""
# Complex sum (usual rules of complex arithmetic, you all know this already)
b = 1 - 1j
print(f'{a} + {b} = {a + b}')

# Increasing the phase angle of a complex number rotates it counter-clockwise
# around the origin in Cartesian space. There are two ways to do this to a
# complex number X (let's say the phase angle is pi/2):
# (i) extract the phase from X, add pi/2 to the phase, and reform the complex
# number, as shown below.
a = 3 + 4j
print('a = ' + str(a) + ' = ' + str(abs(a)) + ' * exp(i' + str(angle(a)) + ')')
print('After adding a phase of pi/2 = {}, this becomes:'.format(pi / 2))
b = abs(a) * exp(1j * (angle(a) + pi / 2))
print('b = ' + str(b) + ' = ' + str(abs(b)) + ' * exp(i' + str(angle(b)) + ')')

# (ii) convert pi/2 into a phase-only complex number, and multiply it by
# the complex number.
a = 3 + 4j
print('a = ' + str(a) + ' = ' + str(abs(a)) + ' * exp(i' + str(angle(a)) + ')')
print('After multiplying by exp(i pi/2) = {}, this becomes:'
      ''.format(exp(1j * pi / 2)))
b = a * exp(1j * pi / 2)
print('b = ' + str(b) + ' = ' + str(abs(b)) + ' * exp(i' + str(angle(b)) + ')')

# Here are two more examples of effecting a rotation in the complex
# plane through multiplication with a phase-only complex value.
a = 3 + 4j
rotation = pi
a_after_rot = a * exp(1j * rotation)
print('a = {}. After rotation by pi:{}'.format(a, a_after_rot))
rotation = -pi / 2
a_after_rot = a * exp(1j * rotation)
print('a = {}. After rotation by -pi/2:{}'.format(a, a_after_rot))

"""
Complex conjugation
"""
# Complex conjugation means changing the sign of the imaginary part
a = 3 + 4j
print('The complex conjugate of', a, 'is', conj(a))
