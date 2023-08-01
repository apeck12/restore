
import numpy as np
from numpy.fft import rfftfreq, fftfreq


def get_mic_relative_freqs(mic, angles=False):
    '''Written by Shawn Zheng
       Use to design filter
    '''
    n_x, n_y = mic.shape
    x, y = np.meshgrid(rfftfreq(n_y), fftfreq(n_x))
    s = np.sqrt(x**2 + y**2)
    s = np.where(s > 0.5, 0.5, s) # cap at 0.5
    return s

class denoise_weight:

    _weight = None
    _highPass = None

    @classmethod
    def calcWeight(cls, mic, f_cent, f_width):
        # exponent 30 is selected that suppresses compenents of 0.125
        # (8 pixel) and beyond to 0.01.
        relative_freqs = get_mic_relative_freqs(mic)
        a = np.log(2) - np.log((np.cos(np.pi * f_width) + 1)) + 1e-30
        a = np.log(100) / a
        weight1 = ((1 - np.cos(relative_freqs / f_cent * np.pi)) / 2) 
        weight2 = ((np.cos((relative_freqs - f_cent) * np.pi * 2) + 1) \
            / 2) ** a
        cls._weight = np.where(relative_freqs < f_cent, weight1, weight2) 
        print("Merge filter: " + str(f_cent) + " " + str(f_width) \
            + " " + str(a))
        #------------------
        cls._highPass = ((1 - np.cos(relative_freqs * np.pi * 2)) / 2) \
            * 0.9 + 0.1
        cls._highPass = cls._highPass ** 0.5

    @classmethod
    def getWeight(cls):
        return cls._weight

    @classmethod
    def getHighPass(cls):
        return cls._highPass
