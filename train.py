#! /usr/bin/env python
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Program for training a convolutional neural network to denoise images 
# from cryogenic electron microscopy (cryo-EM).
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

from h5py import File
import argparse
import sys
import os
from tqdm import tqdm

import numpy as np
from numpy.fft import rfft2
from numpy.fft import irfft2

from pyem import star
from pyem import ctf
from restore.utils import load_star
from restore.utils import load_mic
from restore.utils import bin_mic
from restore.utils import get_patches
from restore.utils import get_mic_freqs
from restore.utils import normalize
from restore.utils import fourier_crop

from restore.model import get_model
from restore.model import load_trained_model
from restore.model import SampleGenerator
from restore.model import Schedule
from restore.model import get_callbacks

from external.memory_saving_gradients import gradients_memory
from keras import backend as K

from input import train_input
import glob

def main(args):
    """Main function for training a denoising CNN"""

    # Training data is stored in an HDF file.
    # If a STAR file is given, the training data will be created
    # If an HDF file is given, the training data will be loaded
    if args.training_mics:
        cutoff_frequency = 1./args.max_resolution
        training_data = args.training_filename
        generate_training_data(args.training_mics, cutoff_frequency, 
                               training_data, args.even_odd_suffix,
                               phaseflip=args.phaseflip)

    elif args.training_data:
        training_data = args.training_data

    else:
        raise Exception(
            "Neither training micrographs or training_data were provided!")

    # Initialize a neural network model for training
    # OR if a pre-trained model is provided, load that instead
    learning_rate = args.learning_rate
    number_of_epochs = args.number_of_epochs
    batches_per_epoch = args.batches_per_epoch
    batch_size = args.batch_size
    
    if args.initial_model:
        nn = load_trained_model(args.initial_model)
    else:
        nn = get_model(learning_rate, layers=args.layers,
                       blocks_per_layer=args.blocks_per_layer)

    # Create the model directory
    model_directory = args.model_directory
    model_prefix = args.model_prefix

    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Set up data generator and callbacks
    data_generator = SampleGenerator(training_data, batch_size)
    
    callbacks = get_callbacks(model_directory, model_prefix, 
                              number_of_epochs, learning_rate)

    # Turn on memory saving gradients
    K.__dict__["gradients"] = gradients_memory
    
    # Train with the 'fit_generator' method from Keras
    history = nn.fit_generator(
                  generator = data_generator,
                  steps_per_epoch = batches_per_epoch,
                  epochs = number_of_epochs,
                  verbose=1,
                  callbacks = callbacks)

    return 

#------------------------------------------------------------------------------
# Read file names in the given directory. Return the list containing all
# MRC file names of full-sum micrographs. Shawn Zheng
#------------------------------------------------------------------------------
def readMicFileNames(micDir):
        aFiles = micDir;
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
                if "_EVN.mrc" in f or "_ODD.mrc" in f:
                        continue
                elif "_DW.mrc" in f:
                        aFullSumFiles.append(f)
        if len(aFullSumFiles) > 0 :
                return aFullSumFiles
        #---------------------------
        for f in aFiles:
                if "_EVN.mrc" in f or "_ODD.mrc" in f:
                        continue
                else:
                        aFullSumFiles.append(f)
        return aFullSumFiles

def changeSuffix(micFileName, newSuffix):
        if "_DW.mrc" in micFileName:
                return micFileName.replace("_DW.mrc", newSuffix)
        else:
                return micFileName.replace(".mrc", newSuffix)


def generate_training_data(training_mics, cutoff, training_data, suffixes,
                           window=192, phaseflip=False):
    """ Generate the training data given micrographs and their CTF information

    Keyword arguments:
    training_mics -- Micrograph STAR file with CTF information for each image
    cutoff -- Spatial frequency for Fourier cropping an image
    training_data -- Filename for the HDF file that is created 

    It is presumed that all images have the same shape and pixel size. 
    Phase-flipping is currently not performed to correct for the CTF.
    """

    aFullSumFiles = readMicFileNames(training_mics)
    n_mics = len(aFullSumFiles)
    #--------------------------
    # cutoff is relative frequence
    #-----------------------------
    apix = 1.0
    #---------
    dset_file = File(training_data, "w")
    dset_shape, n_patches, mic_freqs, mic_angles = get_dset_shape(
                                                       aFullSumFiles, window, 
                                                       apix, cutoff)

    even_dset = dset_file.create_dataset("even", dset_shape, dtype="float32")
    odd_dset = dset_file.create_dataset("odd", dset_shape, dtype="float32")

    orig,even,odd = suffixes.split(",")
    if len(suffixes.split(",")) != 3:
        raise Exception("Improperly formatted suffixes for even/odd mics!")

    print("Pre-processing " + str(n_mics) + " micrographs")
    for i in range(n_mics):
        print("Preprocessing micrograph: " + str(i))
        mic_file = aFullSumFiles[i]
        even_file = changeSuffix(mic_file, "_EVN.mrc")
        odd_file = changeSuffix(mic_file, "_ODD.mrc")
        #--------------------------------------------
        mic_even_patches, apix_bin = process(cutoff, window, 
                                             even_file, mic_freqs, mic_angles,
                                             )

        mic_odd_patches, apix_bin = process(cutoff, window, 
                                            odd_file, mic_freqs, mic_angles
                                            )

        even_dset[i*n_patches: (i+1)*n_patches] = mic_even_patches
        odd_dset[i*n_patches: (i+1)*n_patches] = mic_odd_patches
        

    even_dset.attrs['apix']=apix_bin
    even_dset.attrs['phaseflip']=phaseflip

    odd_dset.attrs['apix']=apix_bin
    odd_dset.attrs['phaseflip']=phaseflip

    dset_file.close()
    return 


def get_dset_shape(micFileNames, window, apix, cutoff_frequency):
    """Calculate the expected shape of the training dataset.
    Returns the shape of the dataset, the number of patches per micrograph,
    and the unbinned spatial frequency and angle arrays so they don't need 
    to be recalculated in later steps"""

    print("*****: " + micFileNames[0])
    first_mic = load_mic(micFileNames[0])
    mic_bin = bin_mic(first_mic, apix, cutoff_frequency)

    s,a = get_mic_freqs(first_mic, apix, angles=True)
    n_patches = len(get_patches(mic_bin, window))
    n_mics = len(micFileNames)

    return (n_patches*n_mics, window, window, 1), n_patches, s, a


def process(cutoff, window, mic_file, freqs, angles, 
            bandpass=True, hp=.005):
    """ Process a training micrograph.

    The following steps are performed:
    (1) The micrograph is loaded, Fourier transformed and Fourier cropped
    (2) A bandpass filter is applied with pass-band from cutoff to 1/200A
    (3) The inverse FT is calculated to return to real-space
    (4) The binned, filtered image is divided into patches, which are
        normalized (Z-score normalization) and returned
    """

    mic = load_mic(mic_file)
    mic_ft = rfft2(mic)

    mic_ft_bin = fourier_crop(mic_ft, freqs, cutoff)
    freqs_bin = fourier_crop(freqs, freqs, cutoff)
    angs_bin = fourier_crop(angles, freqs, cutoff)

    apix_bin = 0.5/freqs_bin[0,-1]

    if bandpass:
        bp_filt = ( (1. - 1./(1.+(freqs_bin/hp)**10)) 
                   + 1./(1.+(freqs_bin/cutoff)**10)/2.)
        mic_ft_bin *= bp_filt

    mic = irfft2(mic_ft_bin).real.astype('float32')
    patches = [normalize(p) for p in get_patches(mic, window)]
    n_patches = len(patches)

    return np.array(patches).reshape((n_patches, window, window, 1)), apix_bin
 

if __name__=="__main__":

    args = train_input.getArgs()
    sys.exit(main(args))
