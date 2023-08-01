#! /usr/bin/env python3
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Utility functions for the restore program.
# These are mainly for IO and image processing including
# Fourier-based image resizing and Fourier filtering. 
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

import mrcfile
import numpy as np
from itertools import product

class SplitInfo:

    def __init__(self):
        self.iImgSizeX = 0
        self.iImgSizeY = 0
        self.iOverlap = 64
        self.patStarts = None
        self.patSizes = None
        self.iNumPatchesX = 0
        self.iNumPatchesY = 0

class Split:

    def __init__(self):
        self.iPatSize = 1024
    
    def doIt(self, img):
        if img.shape[0] <= self.iPatSize or img.shape[1] <= self.iPatSize:
            return self.pad32(img)
        #-------------------------
        splitInfo = SplitInfo()
        splitInfo.iImgSizeY = img.shape[0]
        splitInfo.iImgSizeX = img.shape[1]
        #---------------------------------
        iPatSize = self.iPatSize - splitInfo.iOverlap
        iSizeY = img.shape[0] - self.iPatSize
        iSizeX = img.shape[1] - self.iPatSize
        iRemainX = iSizeX % iPatSize
        iRemainY = iSizeY % iPatSize
        patStartXs = np.arange(0, iSizeX, iPatSize, dtype=np.int)
        patStartYs = np.arange(0, iSizeY, iPatSize, dtype=np.int)
	#--------------------------------------------------------
        patSizeXs = np.full(patStartXs.shape, self.iPatSize)
        patSizeYs = np.full(patStartYs.shape, self.iPatSize)
        patSizeXs[-1] = (patSizeXs[-1] + iRemainX + 31) // 32 * 32
        patSizeYs[-1] = (patSizeYs[-1] + iRemainY + 31) // 32 * 32
        #----------------------------------------------------------
        patStartYs[-1] = img.shape[0] - patSizeYs[-1]
        patStartXs[-1] = img.shape[1] - patSizeXs[-1]
        #--------------------------------------------
        splitInfo.patStarts = [ s for s in product(patStartYs, patStartXs)]
        splitInfo.patSizes = [ s for s in product(patSizeYs, patSizeXs)]
        splitInfo.iNumPatchesX = patStartXs.shape[0];
        splitInfo.iNumPatchesY = patStartYs.shape[0];
        #--------------------------------------------
        patches = []
        for i in range(len(splitInfo.patStarts)):
            start = splitInfo.patStarts[i]
            size = splitInfo.patSizes[i]
            patches.append(img[ start[0]:start[0]+size[0],
                                     start[1]:start[1]+size[1] ])
        return np.array(patches), splitInfo

    def pad32(self, img):
        splitInfo = SplitInfo()
        splitInfo.iImgSizeY = img.shape[0]
        splitInfo.iImgSizeX = img.shape[1]
        splitInfo.iOverlap = 0
        #---------------------
        iPadY = (img.shape[0] + 31) // 32 * 32 - img.shape[0];
        iPadX = (img.shape[1] + 31) // 32 * 32 - img.shape[1];
        iBeforeX, iAfterX = iPadX // 2, iPadX - iPadX // 2
        iBeforeY, iAfterY = iPadY // 2, iPadY - iPadY // 2;
        
        imgPadded = np.pad(img, ((iBeforeY, iAfterY), 
            (iBeforeX, iAfterX)), mode='mean')
        patches = []
        patches.append(imgPadded)
        return np.array(patches), splitInfo

class Assemble:

    def __init__(self):
        return

    def doIt(self, patArray, splitInfo, img):
        if patArray.shape[0] == 1:
            img = self.unpad32(patArray[0], img)
            return img
        #-------------
        mean = np.mean(img)
        img = np.full(img.shape, mean, dtype=np.float)
        #---------------------------------------------
        iHalfOverlap = splitInfo.iOverlap // 2;
        for i in range(patArray.shape[0]):
            iSizeY = splitInfo.patSizes[i][0]
            iSizeX = splitInfo.patSizes[i][1]
            #--------------------------------
            iHalfOverlapX, iHalfOverlapY = 0, 0
            iStartY = splitInfo.patStarts[i][0]
            iStartX = splitInfo.patStarts[i][1] 
            if (i % splitInfo.iNumPatchesX) > 0 :
                iHalfOverlapX = iHalfOverlap
                iStartX += iHalfOverlapX
            if (i // splitInfo.iNumPatchesX) > 0:
                iHalfOverlapY = iHalfOverlap
                iStartY += iHalfOverlapY
            #---------------------------
            iEndX = splitInfo.patStarts[i][1] + iSizeX
            iEndY = splitInfo.patStarts[i][0] + iSizeY
            #-----------------------------------------
            img[iStartY:iEndY, iStartX:iEndX] = \
                patArray[i][iHalfOverlapY:iSizeY, iHalfOverlapX:iSizeX]
            #----------------------------------------------------------
        return img
            
    def unpad32(self, padImg, img):
        iStartX = (padImg.shape[1] - img.shape[1]) // 2
        iStartY = (padImg.shape[0] - img.shape[0]) // 2
        img = padImg[ iStartY:iStartY+img.shape[0], 
            iStartX:iStartX+img.shape[1]]
        return img

