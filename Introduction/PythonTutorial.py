"""Python tutorial to illustrate pointwise multiplication, pointwise
addition, shifting, convolution, and slicing of NumPy arrays.
It also serves to illustrate how to work within the Spyder IDE and
IPython console shipped with Anaconda.

Written to accompany lectures for a module called:
CS356 Signal, Image, and Optical Processing
Thomas J. Naughton, Maynooth University Department of Computer Science
Created: tjn, CS, MU, 15 X 2014
Last updated: tjn, CS, MU, 4 XI 2020

Tested with Python 3.4.3 shipped with Anaconda 2.3.0 (64-bit).

How to run
----------

This file is not meant to be run in its entirety. There are intentional
run-time errors for example, showing the "out of bounds" behaviour of
NumPy indexing. Instead, each line is intended to be run separately, or where
an assignment statement stretches over multiple lines, it is intended
that you would run all of these lines together.

There are few explanations with each of the commands. Instead, you are
required to understand each piece of code from its behaviour with
various inputs.
"""

# This line below is required to be run initially
from numpy import array, arange, roll, convolve


# The = operator means assignment
f = array([1, 2, 1, 2, 1, 3, 5, 6, 7, 1, 3])
h = array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
hr = h[::-1]  # Reverse the list
print(f)
print(f*h)
print(f*hr)

f = array([1, 2, 1, 2,  1,  3])
g = array([1, 0, 1, 0, 10, 10])
f
g
print(f+g)
print(f-g)
print(f*g)

h = arange(6)
h
print(h)
print(roll(h, 1))
print(roll(h, -3))

# Looking at the graphical examples of "convolution" in Wikipedia will
# be sufficient to understand this concept.
print(convolve([1, 2, 3], [0, 1, 0]))
print(convolve([3, 4, 7, 2, 1, 5], [2, 2, 1]))

# This is a classic convolution example. You should make sure to
# understand this input-output pair.
k = array([0, 0, 1, 0, 1, 0, 0])
print(convolve(k, k))

# An important technique for manipulating arrays follows: slicing.
# First, basic indexing of an array
a = array([10, 20, 30, 40, 50])
a
a[0]  # First element
a[2]
len(a)
# a[len(a)]  # Out of bounds
a[len(a)-1]  # Last element
a[-1]  # Shorthand for above

# Next, slicing
a[0:2]  # A slice using two array indices
a[2:len(a)]  # A slice using two array indices
a[2:]  # Shorthand for above

a[2:len(a)-1]  # A slice using two array indices
a[2:-1]  # Shorthand for above

a[2:len(a)-2]  # A slice using two array indices
a[2:-2]  # Shorthand for above

(a[1:-1])[1:-1]  # A slice of a slice
a[1:-1][1:-1]  # We can drop the parentheses

# Each of these is equivalent
print(a)
print(a[0:len(a)])
print(a[0:])
print(a[:len(a)])
print(a[:])

# A third parameter can be used in a slice definition
print(a[::1])  # All elements
print(a[::2])  # All elements with even indices
print(a[::-1])  # All elements in reverse order

# Slicing in two dimensions
data = array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [4, 5, 6]])

print(data)
data[0, :]  # One particular row
data[:, 0]  # One particular column
data[1:, :]  # Include all but the first row
data[:, :-1]  # Include all but the last column

# Indexing a slice based on a condition
a = array([10, 20, 30, 40, 50])
a
# The == operator tests for equality and returns a boolean
a == 30
a[a == 30] = 99
a
(a % 20) == 0  # The modulus operator
a[(a % 20) == 0] = 0  # Just an assignment, so nothing returned
a

# Let's look at an example of this in two dimensions
data = array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [4, 5, 6]])

data
str(data)  # Convert to a string
print(data)
data[data[:, 2] == 3, 1] = 9
print(data)

# If you understand the assignment statement above, then you have
# successfully completed the tutorial. If not, please investigate its
# behaviour by changing the values in that line of code until you do.
