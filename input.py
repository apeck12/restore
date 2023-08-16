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

import argparse

class train_input:
	
    _args = None

    @classmethod
    def getArgs(cls):
        if cls._args != None:
            return cls._args

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Required arguments: 
        # User must provide full path of the directory containing the 
        # training mics (--training_mics, -m)
        # OR training data HDF file (--trainind_data, -t)
        parser.add_argument("--training_mics", "-m", 
            type=str, default=None,
            help="STAR file with micrographs and CTF information")
        parser.add_argument("--training_data", "-t", 
            type=str, default=None,
            help="HDF file containing processed training data")

        # Optional arguments
        parser.add_argument("--even_odd_suffix", "-s", 
            type=str, default="DW,EVN,ODD",
            help="A comma-separated series of three suffixes.  \
                  The first is a suffix in the training micrographs name. \
                  The second is the suffix of the 'even' sums. \
                  The third is the suffix of the 'odd' sums. \
                                                             \
                  If MotionCor2 is used to generate even/odd sums,\
                  the default should be sufficient.")

        parser.add_argument("--max_resolution", "-r", 
            type=float, default=2.0, 
            help="Max resolution to consider in training (angstroms). \
                  Determines the extent of Fourier binning.")

        parser.add_argument("--training_filename", "-f", 
            type=str, default="training_data.hdf",
            help="Name for the newly generated training data file.")

        parser.add_argument("--initial_model", "-i", 
            type=str, default=None,
            help="Initialize training with this pre-trained model")

        parser.add_argument("--batch_size", "-b", 
            type=int, default=10,
            help="Number of training examples used per training batch.")

        parser.add_argument("--learning_rate", "-lr", 
            type=float, default=1e-4,
            help="Initial learning rate for training the neural network")

        parser.add_argument("--number_of_epochs", 
            type=int, default=10,
            help="Number of training epochs to perform. \
                  Model checkpoints are produced after every epoch.")

        parser.add_argument("--batches_per_epoch", 
            type=int, default=50,
            help="Number of training batches per epoch")

        parser.add_argument("--model_prefix", "-x", 
            type=str, default="model", 
            help="Prefix for model files containing the structure and \
                  weights of the neural network.")

        parser.add_argument("--model_directory", "-d", 
            type=str, default="Models",
            help="Directory where trained model files are saved")
    
        parser.add_argument("--phaseflip", dest="phaseflip", 
            action="store_true",
            help="Correct the CTF of the training images by phase-flipping")

        parser.add_argument("--dont_phaseflip", dest="phaseflip", 
            action="store_false",
            help="Don't phase-flip the training images.") 

        parser.add_argument("--layers", dest="layers",
            type=int, default=3,
            help="Number of layers for wide activation U-Net")

        parser.add_argument("--blocks_per_layer", dest="blocks_per_layer",
            type=int, default=4,
            help="Number of blocks per layer for wide activation U-Net")
        
        parser.set_defaults(phaseflip=True) 

        cls._args = parser.parse_args()
        return cls._args

class denoise_input:

    _args = None

    @classmethod
    def getArgs(cls):
        if cls._args != None:
            return cls._args
        
        parser = argparse.ArgumentParser()

        parser.add_argument("--input_micrographs", "-m", 
            type=str, default=None,
            help="Input micrograph directory")

        parser.add_argument("--model", "-p", 
            type=str, default=None,
            help="Neural network model with trained parameters")

        parser.add_argument("--output_suffix", "-s", 
            type=str, default="_denoised",
            help="Suffix added to denoised image output")

        parser.add_argument("--max_resolution", "-r", 
            type=float, default=2.0,
            help="Highest spatial frequencies to consider when denoising \
                  (angstroms). Determines the extent of Fourier binning. \
                  Should be consistent with the resolution of the \
                  training data.")

        parser.add_argument("--merge_resolution", "-x", 
            type=float, default=0.03,
            help="The center of passing band where denoised Fourier \
                  components are merged")

        parser.add_argument("--merge_width", "-w", 
            type=float, default=0.20,
            help="The width of the passing band where denoised Fourier \
                  components are merged. At the cutoff frequent, the  \
                  denoised Fourier component is suppressed to 1 percent \
                  of its original value. The cutoff frequencies are at \
                  central frequence +/- half of the width." )

        parser.add_argument("--dont_merge_noisy", dest="dont_merge_noisy", 
            action="store_true",
            help="Do not merge the low-resolution denoised image with the \
            high-resolution components of the raw image. If false,\
                Otherwise, the merge filter is used as a lowpass filter.")

        parser.add_argument("--denoise_all", dest="denoiseAll",
            action="store_true",
            help="Denoise all MRC files in the input directory.")

        parser.add_argument("--outdir", "-o", dest="outdir",
            type=str, required=False,
            help="Save files to specified output directory instead of input_micrographs dir")
        
        cls._args = parser.parse_args()
        return cls._args

