"""
Transforms very specific to images such as 
color, lighting, contrast, brightness, etc transforms

NOTE: Most of these transforms assume your image intensity
is between 0 and 1, and are torch tensors (NOT numpy or PIL)
"""

import random

import torch as th
from torchvision.transforms.functional import to_tensor
import numpy as np

from ..utils import th_random_choice


class DeNormalize:
    """
    Denormalizes a tensor using provided mean, std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToSqueezedTensor:
    """
    Removes empty dimensions from the mask and converts to a torch.float32 tensor.
    Typically used with B/W masks to remove the "channel" dimension

    :return tensor
    """
    def __init__(self):
        self.to_tensor = MaskToFloatTensor()

    def __call__(self, img):
        # Note, we cannot call the normal torchvision to_tensor method here because it automatically divides by 255 which is NOT what we want.
        return self.to_tensor(img).squeeze()


class MaskPixelsToMap:
    """
    Replaces the pixel values in range [0-255] with class values from supplied value_map.

    :return     : numpy.ndarray with dtype=np.uint8
    """
    def __init__(self, value_map: dict = None):
        """
        :param value_map: Value map to encode. Typically classes are a set of continuous integers starting at 0 (e.g. {55:0, 100:1, 255:2})
        """
        self.value_map = value_map

    def __call__(self, mask):
        """
        :param mask: PIL or OpenCV mask with pixel values in [0-255] range
        :return:
        """
        mask = np.array(mask)  # convert to np
        for k, v in self.value_map.items():
            mask[mask == k] = v  # replace pixels with class values

        return mask.astype(np.uint8)    # make sure it's in UINT8 format


class MaskToTensor:
    """
    Converts a PIL, numpy or CV image to a torch.long representation
    """
    def __call__(self, img):
        return th.from_numpy(np.array(img, dtype=np.int32)).long()


class MaskToFloatTensor:
    """
    Converts a PIL, numpy or CV image to a torch.float32 representation
    """
    def __init__(self, divisor: float = None):
        """
        :param divisor: Optional divisor for the conversion. Can be specified to convert supplied images from [0-255] range to [0.0-1.0]
        """
        self.divisor = divisor

    def __call__(self, img):
        if self.divisor is None:
            return th.from_numpy(np.array(img, dtype=np.float32))
        else:
            return th.from_numpy(np.array(img, dtype=np.float32) / self.divisor)
    

def _blend(img1, img2, alpha):
    """
    Weighted sum of two images

    Arguments
    ---------
    img1 : torch tensor
    img2 : torch tensor
    alpha : float between 0 and 1
        how much weight to put on img1 and 1-alpha weight
        to put on img2
    """
    return img1.mul(alpha).add(1 - alpha, img2)


class Grayscale:

    def __init__(self, keep_channels=False):
        """
        Convert RGB image to grayscale

        Arguments
        ---------
        keep_channels : boolean
            If true, will keep all 3 channels and they will be the same
            If false, will just return 1 grayscale channel
        """
        self.keep_channels = keep_channels
        if keep_channels:
            self.channels = 3
        else:
            self.channels = 1

    def __call__(self, *inputs):
        outputs = []
        idx = None
        for idx, _input in enumerate(inputs):
            _input_dst = _input[0]*0.299 + _input[1]*0.587 + _input[2]*0.114
            _input_gs = _input_dst.repeat(self.channels,1,1)
            outputs.append(_input_gs)
        return outputs if idx >= 1 else outputs[0]


class RandomGrayscale:

    def __init__(self, p=0.5):
        """
        Randomly convert RGB image(s) to Grayscale w/ some probability,
        NOTE: Always retains the 3 channels if image is grayscaled

        p : a float
            probability that image will be grayscaled
        """
        self.p = p

    def __call__(self, *inputs):
        pval = random.random()
        if pval < self.p:
            outputs = Grayscale(keep_channels=True)(*inputs)
        else:
            outputs = inputs
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class Gamma:

    def __init__(self, value):
        """
        Performs Gamma Correction on the input image. Also known as 
        Power Law Transform. This function transforms the input image 
        pixelwise according 
        to the equation Out = In**gamma after scaling each 
        pixel to the range 0 to 1.

        Arguments
        ---------
        value : float
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.value = value

    def __call__(self, *inputs):
        outputs = []
        idx = None
        for idx, _input in enumerate(inputs):
            _input = th.pow(_input, self.value)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]

class RandomGamma:

    def __init__(self, min_val, max_val):
        """
        Performs Gamma Correction on the input image with some
        randomly selected gamma value between min_val and max_val. 
        Also known as Power Law Transform. This function transforms 
        the input image pixelwise according to the equation 
        Out = In**gamma after scaling each pixel to the range 0 to 1.

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range

        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Gamma(value)(*inputs)
        return outputs

class RandomChoiceGamma:

    def __init__(self, values, p=None):
        """
        Performs Gamma Correction on the input image with some
        gamma value selected in the list of given values.
        Also known as Power Law Transform. This function transforms 
        the input image pixelwise according to the equation 
        Out = In**gamma after scaling each pixel to the range 0 to 1.

        Arguments
        ---------
        values : list of floats
            gamma values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.

        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=self.p)
        outputs = Gamma(value)(*inputs)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class Brightness:
    def __init__(self, value):
        """
        Alter the Brightness of an image

        Arguments
        ---------
        value : brightness factor
            =-1 = completely black
            <0 = darker
            0 = no change
            >0 = brighter
            =1 = completely white
        """
        self.value = max(min(value,1.0),-1.0)

    def __call__(self, *inputs):
        outputs = []
        idx = None
        for idx, _input in enumerate(inputs):
            _input = th.clamp(_input.float().add(self.value).type(_input.type()), 0, 1)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]

class RandomBrightness:

    def __init__(self, min_val, max_val):
        """
        Alter the Brightness of an image with a value randomly selected
        between `min_val` and `max_val`

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Brightness(value)(*inputs)
        return outputs

class RandomChoiceBrightness:

    def __init__(self, values, p=None):
        """
        Alter the Brightness of an image with a value randomly selected
        from the list of given values with given probabilities

        Arguments
        ---------
        values : list of floats
            brightness values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=self.p)
        outputs = Brightness(value)(*inputs)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class Saturation:

    def __init__(self, value):
        """
        Alter the Saturation of image

        Arguments
        ---------
        value : float
            =-1 : gray
            <0 : colors are more muted
            =0 : image stays the same
            >0 : colors are more pure
            =1 : most saturated
        """
        self.value = max(min(value,1.0),-1.0)

    def __call__(self, *inputs):
        outputs = []
        idx = None
        for idx, _input in enumerate(inputs):
            _in_gs = Grayscale(keep_channels=True)(_input)
            alpha = 1.0 + self.value
            _in = th.clamp(_blend(_input, _in_gs, alpha), 0, 1)
            outputs.append(_in)
        return outputs if idx >= 1 else outputs[0]

class RandomSaturation:

    def __init__(self, min_val, max_val):
        """
        Alter the Saturation of an image with a value randomly selected
        between `min_val` and `max_val`

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Saturation(value)(*inputs)
        return outputs

class RandomChoiceSaturation:

    def __init__(self, values, p=None):
        """
        Alter the Saturation of an image with a value randomly selected
        from the list of given values with given probabilities

        Arguments
        ---------
        values : list of floats
            saturation values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.

        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=self.p)
        outputs = Saturation(value)(*inputs)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class Contrast:
    """

    """
    def __init__(self, value):
        """
        Adjust Contrast of image.

        Contrast is adjusted independently for each channel of each image.

        For each channel, this Op computes the mean of the image pixels 
        in the channel and then adjusts each component x of each pixel to 
        (x - mean) * contrast_factor + mean.

        Arguments
        ---------
        value : float
            smaller value: less contrast
            ZERO: channel means
            larger positive value: greater contrast
            larger negative value: greater inverse contrast
        """
        self.value = value

    def __call__(self, *inputs):
        outputs = []
        idx = None
        for idx, _input in enumerate(inputs):
            channel_means = _input.mean(1, keepdim=True).mean(2, keepdim=True)
            channel_means = channel_means.expand_as(_input)
            _input = th.clamp((_input - channel_means) * self.value + channel_means,0,1)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]

class RandomContrast:

    def __init__(self, min_val, max_val):
        """
        Alter the Contrast of an image with a value randomly selected
        between `min_val` and `max_val`

        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Contrast(value)(*inputs)
        return outputs

class RandomChoiceContrast:

    def __init__(self, values, p=None):
        """
        Alter the Contrast of an image with a value randomly selected
        from the list of given values with given probabilities

        Arguments
        ---------
        values : list of floats
            contrast values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.

        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=self.p)
        outputs = Contrast(value)(*inputs)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

def rgb_to_hsv(x):
    """
    Convert from RGB to HSV
    """
    hsv = th.zeros(*x.size())
    c_min = x.min(0)
    c_max = x.max(0)

    delta = c_max[0] - c_min[0]

    # set H
    r_idx = c_max[1].eq(0)
    hsv[0][r_idx] = ((x[1][r_idx] - x[2][r_idx]) / delta[r_idx]) % 6
    g_idx = c_max[1].eq(1)
    hsv[0][g_idx] = 2 + ((x[2][g_idx] - x[0][g_idx]) / delta[g_idx])
    b_idx = c_max[1].eq(2)
    hsv[0][b_idx] = 4 + ((x[0][b_idx] - x[1][b_idx]) / delta[b_idx])
    hsv[0] = hsv[0].mul(60)

    # set S
    hsv[1] = delta / c_max[0]

    # set V - good
    hsv[2] = c_max[0]

    return hsv
