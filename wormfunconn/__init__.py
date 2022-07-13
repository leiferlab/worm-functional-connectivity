__all__ = ['integral','integral_py','convolution1','convolution',
           'FunctionalAtlas','exp_conv_2','exp_conv_2b']

__version__ = "0.1.9.4"

strains = ["wild-type", "unc-31"]

from ._integration import integral
from .integration import integral as integral_py
from ._convolution import convolution1, convolution
from .FunctionalAtlas import FunctionalAtlas
from .ExponentialConvolution_min import ExponentialConvolution_min
from .resp_fun import exp_conv_2, exp_conv_2b
from .website_text import website_text
