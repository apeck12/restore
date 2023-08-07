#! /usr/bin/env python
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Program denoising cryo-EM images with a trained convolutional network 
# See README and help text for usage information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import sys
import os
import argparse
from tqdm import tqdm

import numpy as np
from numpy.fft import rfft2, irfft2

from pyem import star
from pyem import ctf
from restore.utils import load_star
from restore.utils import load_mic
from restore.utils import save_mic
from restore.utils import bin_mic
from restore.utils import get_mic_freqs
from restore.utils import normalize
from restore.utils import fourier_crop
from restore.utils import fourier_pad_to_shape
from restore.utils import next32
from restore.utils import smoothstep

from restore.model import load_trained_model

from input import denoise_input
from weighting import denoise_weight
from restore import split_image
from restore.utils import get_mic_relative_freqs
import glob

#------------------------------------------------------------------------------
# Read file names in the given directory. Return the list containing all
# MRC file names of full-sum micrographs. Shawn Zheng
#------------------------------------------------------------------------------
def readMicFileNames(micDir, denoiseAll):
    aFiles = micDir
    if aFiles[len(aFiles) - 1] == '/':
        aFiles = aFiles + "*.mrc"
    else:
        aFiles = aFiles + "/*.mrc"
    #----------------------------------
    aFiles = glob.glob(aFiles)
    if len(aFiles) == 0:
        return
    #--------------
    aFullSumFiles = []
    for f in aFiles:
        if "_denoised" in f:
            continue        
        elif "_EVN.mrc" in f or "_ODD.mrc" in f:
             if denoiseAll == False:                                      
                continue
        aFullSumFiles.append(f)
    return aFullSumFiles


def main(args):
    """ Main denoising CNN function """

    # Load STAR file and neural network
    aMicFiles = readMicFileNames(args.input_micrographs, args.denoiseAll)
    num_mics = len(aMicFiles)
    apix = 1.0
    cutoff_frequency = 1./args.max_resolution  
    nn = load_trained_model(args.model)
    suffix = args.output_suffix
    merge_noisy = not args.dont_merge_noisy
    merge_freq1 = 1./(args.merge_resolution+args.merge_width)
    merge_freq2 = 1./args.merge_resolution


    # Main denoising loop
    for i in range(num_mics):
        print("Denoise micrograph: " + str(i))
        mic_file = aMicFiles[i]

        # Pre-calculate frequencies, angles, and soft mask
        if not i:
            first_mic = load_mic(mic_file)
            freqs, angles = get_mic_freqs(first_mic, apix, angles=True)    
            denoise_weight.calcWeight(first_mic, args.merge_resolution, \
                args.merge_width)

        new_mic = process(nn, mic_file, freqs, angles, apix, cutoff_frequency, merge_noisy, outdir=args.outdir) 
        new_mic_file = mic_file.replace(".mrc", "{0}.mrc".format(suffix))
        save_mic(new_mic, new_mic_file)

    return

def process(nn, mic_file, freqs, angles, apix, cutoff, merge_noisy=True, outdir=None):
    """ Denoise a cryoEM image 
 
    The following steps are performed:
    (1) The micrograph is loaded, phaseflipped, and Fourier cropped
    (2) A bandpass filter is applied with pass-band from cutoff to 1/200A
    (3) The inverse FT is calculated to return to real-space
    (4) The micrograph is padded do a dimension divisible by 32
    (5) The padded is passed through the CNN to denoise then unpadded
    (6) The micrograph is upsampled by padding the Fourier transform 
        with zeros. This procedure creates a sharp edge between components
        with finite amplitudes and zero amplitude, which manifests as
        high-frequency 'ringing' artefacts in real space. To reduce these,
        we apply a soft mask to the denoised FT.
    (7) Optional: the low-pass filtered denoised image is combined with
        the complementary high-resolution noisy image. 
    """

    # Load the micrograph and phase-flip to correct the CTF
    mic = normalize(load_mic(mic_file))
    mic_ft = rfft2(mic) 

    # Fourier crop the micrograph and bandpass filter
    mic_ft_bin = fourier_crop(mic_ft, freqs, cutoff)
    freqs_bin = fourier_crop(freqs, freqs, cutoff)
    #---------------------------------------------
    mic_bin = normalize(irfft2(mic_ft_bin).real)
    #-------------------------------------------
    
    split = split_image.Split()
    patches, split_info = split.doIt(mic_bin)
    
    for i in range(patches.shape[0]):
        p_x = patches[i].shape[0]
        p_y = patches[i].shape[1]
        denoised = nn.predict(patches[i].reshape((1, p_x, p_y, 1)))
        patches[i] = denoised.reshape(p_x, p_y)

    assemble = split_image.Assemble()
    denoised = assemble.doIt(patches, split_info, mic_bin)
    
    #-----------------------------------------------------
    # Upsample by Fourier padding
    denoised_ft = rfft2(normalize(denoised))
    denoised_ft_full = fourier_pad_to_shape(denoised_ft, mic_ft.shape)

    if outdir is not None:
        savename = os.path.join(outdir, mic_file.split("/")[-1])
        savename = savename.replace(".mrc", "{0}.npy".format("_denoised_ft"))
        np.save(savename, denoised_ft_full)
        savename = os.path.join(outdir, mic_file.split("/")[-1])
        savename = savename.replace(".mrc", "{0}.npy".format("_original_ft"))
        np.save(savename, mic_ft)

    #-----------------------------------------------------------------
    if merge_noisy:
        weight = denoise_weight.getWeight() * 0.25
        denoised_ft_full = denoised_ft_full * weight  + mic_ft * (1 - weight)
        #--------------------------------------------------------------------
        highPass = denoise_weight.getHighPass()
        #print(highPass)
        #denoised_ft_full = denoised_ft_full * highPass

    denoised_full = irfft2(denoised_ft_full).real.astype(np.float32)
    new_mic = denoised_full
    return new_mic


if __name__ == "__main__":
    args = denoise_input.getArgs() 
    sys.exit(main(args))
