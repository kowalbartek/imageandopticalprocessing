"""imageutils - a module for manipulating images stored as NumPy ndarrays

(C) Thomas J. Naughton, Maynooth University Department of Computer Science,
Maynooth, County Kildare, Ireland
tomn@cs.nuim.ie
http://www.cs.nuim.ie/~tomn
Created: tjn, CS, MU, 1 XII 2014, first version of window_2d (padding only)
Modified:
tjn, 20 XII 2014, ndarray_to_str, str_to_ndarray, floor_even
tjn, 26 XII 2014, filled_regular_polygon (special cases)
tjn, 15 III 2015, filled_regular_polygon (all convex shapes)
tjn, 16 III 2015, window_2d (allow cropping), shift function
tjn, 20 V 2015, _check_numeric_array and related functions
tjn, 22 V 2015, masks_concentric, partition_concentric functions
tjn, 27 V 2015, disc function
tjn, 18 VI 2015, nrms_error function
tjn, 13 X 2015, stripped out unnecessary functionality for CS356
tjn, 29 X 2015, imread_sc, imsave_sc, create_animated_gif functions
tjn, 16 XI 2015, ignore low contrast image warnings in imsave_sc()
tjn, 19 XI 2015, additional arguments to window_2d()
tjn, 26 XI 2015, roll_2d function
tjn, 13 II 2017, ensure pad() receives only integers as `pad_width`
tjn, 26 VI 2017, window_2d can crop one dim while padding the other
tjn, 22 X 2020, using a custom version of `rescale_intensity`

Tested with Anaconda using Python 3.6.
"""

import warnings
import subprocess

from scipy import (pad, isscalar, ndarray, array, zeros, ogrid, rint, uint8,
                   ceil, roll)

from skimage import util, io

# from skimage.exposure import rescale_intensity
from skimage_exposure import rescale_intensity


def _is_numeric_scalar(a, min_val=None):
    """Returns True if 'a' is a numeric scalar.

    Numeric scalars include int, float, bool.
    """
    if min_val is None:
        min_val = a
    return isscalar(a) and not isinstance(a, str) and (a >= min_val)


def _ensure_int(a, min_val=None):
    """Ensure argument is suitable for convertion to an int, and convert it.

    Specifically, this function guards against the string '5' being
    interpreted as an int as would be the case when calling int('5').
    """
    if _is_numeric_scalar(a, min_val):
        return int(a)
    else:
        raise ValueError('Argument should be a numeric scalar.')


def _ensure_numeric_array(a,
                          shape=None,
                          num_dims=None,
                          min_val=None,
                          lbound=None,
                          order=None):
    """Ensure that argument is a nonempty ndarray of numeric scalars.

    Lists and tuples are allowed as argument 'a'. The user is responsible for
    ensuring that the variable 'a' can be converted using np.array().
    See function _check_numeric_array() for an explanation of other arguments.

    A (modified, if necessary) ndarray is returned.
    """
    if isinstance(a, (list, tuple)):
        a = array(a)
    _check_numeric_array(a,
                         shape=shape,
                         num_dims=num_dims,
                         min_val=min_val,
                         lbound=lbound,
                         order=order)
    return a


def _check_numeric_array(a,
                         shape=None,
                         num_dims=None,
                         min_val=None,
                         lbound=None,
                         order=None):
    """Check that argument is a nonempty ndarray of numeric scalars.

    Arguments:
    'a' is the variable to be tested.
    'shape' is an optional shape that 'a' should have.
    'num_dims' is the number of dimensions that 'a' should have. (Note, a
        ndarray with shape (3,) is a 1D array, and a ndarray with shape (3, 1)
        is a 2D array even though it has one singleton dimension. MATLAB/GNU
        Octave does not make such a distinction.)
    'min_val' is the minimum numerical value each value in 'a' should have.
    'lbound' a lower bound that each value in 'a' should be strictly greater
        than.
    'order' an ordering on the data where '>' means strictly increasing.
        Arrays are ordered according to their ndarray.flat iterator.

    If the argument is not of the correct type, an error is raised.
    """

    def _strictly_increasing(L):
        # From Andrea Griffini (user 6502) via http://stackoverflow.com/
        # questions/4983258/python-how-to-check-list-monotonicity
        return all(x < y for x, y in zip(L, L[1:]))

    """
    Functionality begins here
    """
    # Check argument type and check that it is nonempty
    if (not isinstance(a, ndarray)) or (a.size == 0):
        raise TypeError('Argument should be a nonempty ndarray.')
    # Check shape if appropriate
    if (shape is not None) and a.shape != shape:
        raise ValueError('Argument should be a ndarray with shape ' +
                         str(shape) + '.')
    # Check dimensions if appropriate
    if (num_dims is not None) and len(a.shape) != num_dims:
        raise ValueError('Argument should be a ndarray with exactly ' +
                         str(num_dims) + ' dimensions.')
    # Check type of values in array
    first_val = next(a.flat)
    if not _is_numeric_scalar(first_val):
        raise ValueError('Argument should be an ndarray of numeric scalars.')
    # Check if each element of 'a' is >= the minimum value, if appropriate
    # (at this stage we know each element of 'a' is a numeric scalar).
    if min_val is not None:
        if not _is_numeric_scalar(min_val):
            raise ValueError('Argument should be a numeric scalar.')
        elif min(a.flat) < min_val:
            raise ValueError('Argument should have values >= ' +
                             str(min_val) + '.')
    # Check if each element of 'a' is > the lower bound, if appropriate
    # (at this stage we know each element of 'a' is a numeric scalar).
    if lbound is not None:
        if not _is_numeric_scalar(lbound):
            raise ValueError('Argument should be a numeric scalar.')
        elif min(a.flat) <= lbound:
            raise ValueError('Argument should have values > ' + str(lbound) +
                             '.')
    # Check if each element of 'a' is ordered, if appropriate
    # (at this stage we know each element of 'a' is a numeric scalar).
    if order is not None:
        if order == '>':
            if not _strictly_increasing(a.flat):
                raise ValueError('Argument should be a list of strictly ' +
                                 'increasing numbers.')
        else:
            raise ValueError("Unrecognised argument '" + str(order) + "'.")


def _ensure_pair_numeric_array(a, min_val=None, lbound=None):
    """Ensure that argument is a pair of numeric scalars in a ndarray.

    If a scalar is passed, use it for each element of the pair. Lists/tuples
    are allowed. A (modified, if necessary) ndarray is returned.
    """
    if isscalar(a):
        a = a, a
    return _ensure_numeric_array(a,
                                 shape=(2,),
                                 min_val=min_val,
                                 lbound=lbound)


def _check_2d_numeric_array(a, min_val=None, order=None):
    """Check that argument is a nonempty 2D ndarray of numeric scalars.
    If the argument is not of the correct type, an error is raised.
    """
    _check_numeric_array(a, num_dims=2, min_val=min_val, order=order)


def imread_sc(fname, as_gray=True):
    """Read an image file from disk and rescale to the range [0, 1].

    This function reads an image file, converts its uint8 values to floats,
    and rescales to the range [0, 1], stretching the range as much as
    possible.

    rescale_intensity(img_as_float()) is explicitly required because if the
    image is already a greyscale image, it will not be converted from uint to
    float and will not be rescaled.

    The as_gray property has the same meaning as that argument in imread(),
    except it has a different default value.

    As an example, the stretching property implies that if the image loaded
    from disk is ndarray([[100, 120], [110, 115]], dtype=uint8) then the value
    returned from this function will be ndarray([[0., 1.], [0.5, 0.75]]).

    If an exception is thrown, just pass it directly to the caller.
    """
    return rescale_intensity(util.img_as_float(io.imread(fname)))


def imsave_sc(fname, im):
    """Save a real-valued image as an 8-bit depth image.

    This function rescales the image if necessary, converts it to uint8
    values, and writes to an image file.

    Images are stretched and rescaled to the [0, 1] range if they have any
    values outside this range.

    It works for both greyscale and colour images.

    If an exception is thrown, just pass it directly to the caller.
    """
    if (im > 1).any() or (im < 0).any():
        im = rescale_intensity(im, out_range=(0, 1))
    # Convert to integers in the range [0, 255] before saving. Ignore warnings
    # related to low contrast images (we have legitimate reasons to write out
    # completely black frames as images). These warnings are only ignored
    # within the context of catch_warnings() (i.e. within the scope of the
    # 'with' statement).
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                                message='.*is a low contrast image.*',
                                category=UserWarning)
        io.imsave(fname, rint(im * 255).astype(uint8))


def window_2d(im, win, shift=(0, 0), new_val=0, rel_shift=None):
    """A function for padding, cropping, pasting 2D NumPy arrays.

    Pad image im up to shape 'win' while keeping im centred.
    Crop image 'im' to shape 'win' by removing (by default) an equal number of
    rows and columns from each end.

    If an odd number of rows/columns are to be padded/cropped then do less
    at left & above and do more at right & below.

    Arguments
    ---------
    im : 2D ndarray
        Image to be manipulated.
    win : integer pair; integer scalar
        Desired shape of returned image. If a scalar, then assume it
        represents square side length.
    shift : integer pair (a_y, a_x); integer scalar
        Vertical/horizontal direction shift of window (in pixels) from centre
        of im. A positive a_y and a_x shifts window downwards and rightwards,
        respectively. If a scalar, assume an identical shift in each
        dimension. If rel_shift is not None, then ignore this argument.
    new_val : complex scalar
        Values used to fill pixels created through padding.
    rel_shift : real-valued pair; real-valued scalar
        Scalars in range [-1, 1] denote relative position, and are used to
        specify the absolute maximum shift in any direction while ensuring
        the window and image overlap everywhere.
        As such, (-1, -1) denotes top left, (1, 1) denotes bottom right,
        (0, 1) denotes centre right, and so on.
        If a scalar, assume an identical shift in each dimension.
        If this argument is not None, then the shift argument is ignored.
    """
    # Raise an error if 'im' is not a 2D ndarray of numeric scalars. Empty 2D
    # arrays are allowed.
    if not (isinstance(im, ndarray) and im.shape == (0, 0)):
        _check_2d_numeric_array(im)

    # Deal with the rel_shift argument. Differentiate between 0 and None.
    if rel_shift is not None:
        # Ensure rel_shift is a pair of scalars, if only one scalar provided
        if isscalar(rel_shift):
            rel_shift = rel_shift, rel_shift
        # Ensure rel_shift is a pair of floats, in an ndarray
        rel_shift = array([float(a) for a in rel_shift])
        # Ensure the values are in the range [-1, 1]
        rel_shift[rel_shift > 1] = 1
        rel_shift[rel_shift < -1] = -1
        # Determine the absolute shift in pixels corresponding to the relative
        # shift 'rel_shift'. Overwrite the values in 'shift' (if any).
        shift = rel_shift * abs(im.shape - win) // 2

    # Ensure 'win' and 'shift' each describe a 2D array consisting of exactly
    # two numeric scalars (if a scalar is passed, use it for each dimension).
    win = _ensure_pair_numeric_array(win)
    shift = _ensure_pair_numeric_array(shift)
    # Convert to type int (because truncated floats are not sufficient for
    # function `pad`)
    win = win.astype(int)
    shift = shift.astype(int)

    # There are four cases to consider: no operations required, cropping
    # each dimension, padding each dimension, and a different operation for
    # each dimension.
    # The case where the window has an equal shape to the input
    if (win == im.shape).all():
        # Return im unchanged, but deal with a nonzero shift argument
        if not (shift == 0).all():
            raise ValueError('window_2d() does not know how to deal with a ' +
                             'nonzero shift argument when the window and ' +
                             'input have the same shape.')
        return im

    # The case where the window is smaller than the input
    elif (win <= im.shape).all():
        # Calculating the coordinates of the cropped im
        #        ystart = (im.shape[0] - win[0]) // 2
        #        xstart = (im.shape[1] - win[1]) // 2
        #        yend = ystart + win[0]
        #        xend = xstart + win[1]
        start = (im.shape - win) // 2 + shift
        end = start + win
        return im[start[0]:end[0], start[1]:end[1]]

    # The case where the window is larger than the input
    elif (win >= im.shape).all():
        # Calculating the padding required on each side of im
        #        ybefore = (shape[0] - im.shape[0]) // 2
        #        yafter = shape[0] - im.shape[0] - ybefore
        #        xbefore = (shape[1] - im.shape[1]) // 2
        #        xafter = shape[1] - im.shape[1] - xbefore
        before = (win - im.shape) // 2 + shift
        after = win - im.shape - before
        return pad(im,
                   #    ((before[0], after[0]), (before[1], after[1])),
                   list(zip(before, after)),
                   mode='constant',
                   constant_values=new_val)

    # The case where the window is larger than the input in one dimension and
    # smaller than the input in the other dimension.
    else:
        # Do the required processing one dimension at a time
        im = window_2d(im,
                       (win[0], im.shape[1]),
                       shift=shift,
                       new_val=new_val,
                       rel_shift=rel_shift)
        im = window_2d(im,
                       (im.shape[0], win[1]),
                       shift=shift,
                       new_val=new_val,
                       rel_shift=rel_shift)
        return im


def roll_2d(im, shift):
    """Perform two successive orthogonal roll() operations.

    Each element of shift is passed directly to SciPy's roll().
    If a scalar, assume an identical shift for each dimension.
    """
    # Ensure 'shift' describes a 2D array consisting of exactly two numeric
    # scalars (if a scalar is passed, use it for each dimension).
    shift = _ensure_pair_numeric_array(shift)
    return roll(roll(im, shift[0], axis=0), shift[1], axis=1)


def shift(im, pixels, new_val=0):
    """Shift a 2D ndarray laterally, padding each new pixel with a constant.

    Argument im is a 2D ndarray.
    Argument pixels is a pair (pixels_downwards, pixels_rightwards), where a
    negative value for either means shift in the opposite direction.

    An image with the same shape as im is returned.

    This function differs from roll() in that the pixels shifted outside the
    extent of im are lost, rather than circularly shifted to the other end
    of the ndarray as with roll().
    """
    # Ensure a NumPy array is passed for im
    if not isinstance(im, ndarray):
        raise TypeError('First argument must be of type ndarray.')
    # Ensure pixels gives a value for each dimension
    pixels = array(pixels)
    if pixels.shape != (2,):
        raise TypeError('shift() requires a shift to be specified for ' +
                        'exactly two dimensions.')
    # Ensure pixels contains only ints or truncated floats
    if (pixels.astype(int) == pixels).all():
        # Convert to type int
        pixels = pixels.astype(int)
    else:
        raise ValueError('shift() requires shift values to be integers.')

    # Perform padding followed by cropping in vertical dimension
    if pixels[0] > 0:
        im = pad(im,
                 ((pixels[0], 0), (0, 0)),
                 'constant',
                 constant_values=new_val)
        im = im[:-pixels[0], :]
    elif pixels[0] < 0:
        im = pad(im,
                 ((0, -pixels[0]), (0, 0)),
                 'constant',
                 constant_values=new_val)
        im = im[-pixels[0]:, :]
    else:
        # Do nothing
        pass

    # Perform padding followed by cropping in horizontal dimension
    if pixels[1] > 0:
        im = pad(im,
                 ((0, 0), (pixels[1], 0)),
                 'constant',
                 constant_values=new_val)
        im = im[:, :-pixels[1]]
    elif pixels[1] < 0:
        im = pad(im,
                 ((0, 0), (0, -pixels[1])),
                 'constant',
                 constant_values=new_val)
        im = im[:, -pixels[1]:]
    else:
        # Do nothing
        pass

    return im


def shift_r(im, pixels):
    """Wrapper for shift(), to make it as convenient to call as roll().
    """
    return shift(im, (0, pixels))


def disc(diameter, shape=None, centre=None, dtype=bool):
    """Return a mask with shape 'shape' and containing a disc with diameter
    'diameter' centred at array indices 'centre', and containing values of
    type 'dtype'.

    By convention, and so that each disc will have a well defined central
    pixel, each disc has odd diameter. When 'shape' has even dimensions, and
    'centre' == None, the disc's central pixel will be the bottom right pixel
    of the central four pixels. I.e. when 'shape' is (2, 2) then the central
    pixel is at index [1,1].
    """
    # Ensure that 'diameter' is valid
    diameter = _ensure_int(diameter)  # , min_val=0) #, min_val=1)
    # Determine disc radius
    if diameter <= 0:
        diameter = 0
        radius = 0
    else:
        radius = (diameter - 1) // 2

    # Ensure that 'shape' is valid
    if shape is None:
        # If 'shape' is not specified, take its value from 'diameter'
        shape = array((diameter, diameter))
    else:
        # Ensure shape is a pair of nonzero numeric scalars. Allow a scalar
        # to be passed for 'shape' to indicate a square.
        shape = _ensure_pair_numeric_array(shape)  # , min_val=diameter)

    # Ensure that 'centre' is valid
    if centre is None:
        # Set central indices to be central indices of 'shape'
        y, x = shape // 2
    else:
        # Ensure pair of central indices is a pair of numeric scalars (do not
        # allow a single scalar value).
        y, x = _ensure_numeric_array(centre, shape=(2,))

    if diameter == 0:
        # Take care of the special case of a non-existent disc. Note, this
        # subltly different from testing for radius == 0, which would have
        # inadvertantly caught diameter == 1 and diameter == 2.
        return zeros(shape, dtype=bool)
    else:
        # Create two orthogonal 1D arrays describing vertical and horizontal
        # coordinates respectively.
        v, h = ogrid[-y:(shape[0] - y), -x:(shape[1] - x)]

        # Broadcast the orthogonal 1D arrays to define a 2D array. All
        # coordinates whose hypothenuse is <= the radius will be inside the
        # disc.
        return ((v ** 2 + h ** 2) <= radius ** 2).astype(dtype)


def create_animated_gif(fname,
                        start=0,
                        stop=1,
                        delay=100,
                        out_fname=None,
                        quiet_on_success=False):
    """Create an animated GIF from a list of image filenames.

    This function requires the ImageMagick package to be installed which
    includes the "convert" command.

    If fname is a string, then it is assumed that the list of filenames is
    (fname+str(start)+'.png', fname+str(start+1)+'.png', ...,
     fname+str(stop-1)+'.png').
    If fname is a list/tuple/generator (assumed to be a list/tuple/generator
    of filenames, each including a file extension) then start and stop are
    ignored.

    delay specifies a delay (in milliseconds) between frames in the animated
    GIF.

    The filename for the animated GIF will be out_fname+'.gif' (if out_fname
    is not None) or else it will be fname+'.gif' (if fname is a string) or
    else it will be fnames[0]+'_anim.gif'.

    quiet_on_success is a flag used to supress printing a status message is
    the call to ImageMagick's convert is successful.
    """
    # The ImageMagick "convert" command as recognised by the operating system
    convert_cmd = 'convert'

    # Ensure fname is a list of filename strings
    if isinstance(fname, str):
        fnames = [fname + str(a) + '.png' for a in range(start, stop)]
    else:
        # Convert to a list in case it is a generator
        fnames = list(fname)
    # Construct the output filename if needed
    if out_fname is None:
        if isinstance(fname, str):
            out_fname = fname
        else:
            out_fname = fnames[0] + '_anim'
    # Convert the delay from units of milliseconds into units of centiseconds
    # as expected by ImageMagick.
    delay = int(ceil(delay / 10))
    # Create a list of tokens that will be separated by spaces when used as
    # command line arguments.
    out_fname += '.gif'
    args = ["-delay", str(delay)] + fnames + [out_fname]
    # Call a program external to Python
    try:
        retcode = subprocess.check_call([convert_cmd] + args)
        if not quiet_on_success:
            print('Animated GIF "' + out_fname + '" created.')
    except FileNotFoundError:
        retcode = -1
        print('File not found error executing command "' + convert_cmd +
              '". Have you installed ImageMagick?')
    except subprocess.CalledProcessError as e:
        retcode = e.returncode
        print('Unable to create file "' + out_fname +
              '" (ImageMagick returned error code: ' + str(retcode) + ').')
    return retcode
