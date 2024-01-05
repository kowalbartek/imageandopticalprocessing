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
Created:
tjn, CS, NUIM, 1 X 2012
Modified:
tjn, CS, MU, 29 XI 2014, converted from GNU Octave to Python
tjn, CS, MU, 13 X 2015, more example images at end of sheet
tjn, CS, MU, 22 X 2020, adapted to run on PythonAnywhere

Tested with Anaconda using Python 3.6.

Execution note: This code is not designed to run in its entirety in one go.
Instead, it is designed to be highlighted piece by piece in Spyder and
executed by pressing F9.
"""

import matplotlib
from spatialfilteringdemo import (spatial_filtering_demo,
                                  enhance_focussed_parts_of_colour_image)
from quickfunctions import quick_close
import quickfunctions

#quickfunctions.FONT_SIZE = 7
#quickfunctions.SUP_FONT_SIZE = 10
matplotlib.rcParams.update({'font.size': 7})
matplotlib.rcParams.update({'savefig.dpi': 300})
matplotlib.rcParams['figure.figsize'] = [24, 16]
#matplotlib.rcParams['figure.dpi'] = 100

# Look at the source code or type the command below in the IPython console
# to understand the parameters of the function spatial_filtering_demo()
# Note: this command only works in the IPython console (e.g. pressing F9
# here); it will generate a syntax error if run as part of a script.
# Type this into IPython for help about the function:
# spatial_filtering_demo?


"""
Using the image seen several times already in lectures...
"""
# Enhancing all edges
a = spatial_filtering_demo('sampleshapes.bmp', 'freq', 0.2, 0)

# Enhancing horizontal edges
a = spatial_filtering_demo('sampleshapes.bmp', 'orient', 0, 1)

# Enhancing diagonal edges in one direction
a = spatial_filtering_demo('sampleshapes.bmp', 'orient', 45, 1)

# Removing diagonal edges in one direction (ringing evident)
a = spatial_filtering_demo('sampleshapes.bmp', 'orient', 45, 0)

# Removing all edges (ringing evident)
a = spatial_filtering_demo('sampleshapes.bmp', 'freq', 0.2, 1)

# Removing regions of constant intensity
a = spatial_filtering_demo('sampleshapes.bmp', 'freq', 0.02, 'hp')

# Extreme blurring
a = spatial_filtering_demo('sampleshapes.bmp', 'freq', 0.02, 'lp')

# An identity transform -- no modification of input except for rounding errors
# (if any).
a = spatial_filtering_demo('sampleshapes.bmp', 'freq', 0, 'hp')


"""
Using a different image...
"""
quick_close()
# Removing regions of constant intensity (this is not just making the image
# darker: note that although the background and some pegs are darker, the
# edges should be equally visible in the filtered image).
a = spatial_filtering_demo('pegs.png', 'freq', 0.009, 0)

# Gives rise to "ringing effects" due to sharp edge of filter
a = spatial_filtering_demo('pegs.png', 'freq', 0.1, 1)

# Edges at one orientation
a = spatial_filtering_demo('pegs.png', 'orient', 0, 1)

# Ringing in one direction only
a = spatial_filtering_demo('pegs.png', 'orient', 0, 0)

"""
Isolating in-focus regions of an image
"""
quick_close()
# Applying a single band-pass filter
enhance_focussed_parts_of_colour_image('green.jpg', 0.05)
# Combining the results of several band-pass filters
enhance_focussed_parts_of_colour_image('green.jpg', (0.05, 0.07, 0.09, 0.1))
# Another image
enhance_focussed_parts_of_colour_image('mcq.jpg', 0.4)
# Another image
enhance_focussed_parts_of_colour_image('mcq2.jpg', 0.3)
# Another image, enhancing very fine edges only
enhance_focussed_parts_of_colour_image('yellow.jpg', 0.3)
# The previous image, enhancing gross edges
enhance_focussed_parts_of_colour_image('yellow.jpg', 0.1)
# The previous image, enhancing more parts of image than edges
enhance_focussed_parts_of_colour_image('yellow.jpg', 0.01)
