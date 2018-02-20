from __future__ import print_function

import cv2
import numpy as np
from helper import generateMotionBlurPSF
import os

class DataSource(object):
    def __init__(self, imageSource, path='', length=[0,15], orientation=[0,180], filterCount=2000, minSize=100):
        print('Generating filters.')
        self.generateFilters(length, orientation, filterCount)
        print('Generating filters --- DONE.')

        self.minSize = minSize + self.filters[0].shape[2]
        print('Reading images.')
        self.readImages(imageSource, path, self.minSize)
        print('Reading images --- DONE.')
                
        
    def readImages(self, imageSource, path, minSize):
        self.images = []
        with open(imageSource, 'r') as f:
            for line in f:
                line = line.strip()
                try:
                    newImage = cv2.imread(os.path.join(path, line)).astype(np.float32)
                    newImage /= 256.0
                    newImage *= 0.8
                    newImage += 0.1
                    if len(newImage) == 2:
                        newImage = np.expand_dims(newImage, axis=2)
                        newImage = np.repeat(newImage, 3, axis = 2)
                    if newImage.shape[0] > minSize and newImage.shape[1] > minSize:
                        self.images.append(newImage.transpose(2,0,1))
                    else:
                        print('Warning: Image is too small "{}".'.format(line))
                except:
                    print('ERROR: While reading image "{}".'.format(line))

                    
    def generateFilters(self, length, orientation, filterCount):
        self.filters = []
        for i in range(filterCount):
            #o = (orientation[1] - orientation[0]) * float(i) / filterCount # 
            o = (orientation[1] - orientation[0])* np.random.ranf() + orientation[0]
            l = (length[1] - length[0]) * np.random.ranf() + length[0]
            #l = length[1] #
            psf = generateMotionBlurPSF(o, l)
            border = int((length[1] - psf.shape[0]) / 2)
            psf = np.pad(psf, [(border,border), (border,border)], mode='constant')
            psf = np.expand_dims(psf, axis=0)
            psf = np.repeat(psf, 3, axis = 0)
            self.filters.append(psf)
        self.filters = np.stack(self.filters, axis=0)
        
                    
    def getBatch(self, count=32, cropSize=100):
        
        cropSize = cropSize + self.filters[0].shape[2]
        
        idx = np.random.choice(len(self.images), count)
        images = [self.images[i] for i in idx]
        outImages = []
        for image in images:
            i1 = np.random.randint(image.shape[1] - cropSize)
            i2 = np.random.randint(image.shape[2] - cropSize)
            outImages.append(image[:, i1:i1+cropSize, i2:i2+cropSize])
        data = np.stack(outImages)
        
        idx = np.random.choice(self.filters.shape[0], count)
        #idx = np.arange(self.filters.shape[0])
        psf = self.filters[idx]
        
        return data, idx, psf