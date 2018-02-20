from __future__ import print_function
import cv2
import numpy as np
import threading
import sys
import os
import math

#import matplotlib.pyplot as plt
from pandas import ewma
from matplotlib.collections import patchstr

class ReaderThread(threading.Thread):
    def __init__(self, imageQueue, fileList, imageDir, scaleFactor=1.0):
        threading.Thread.__init__(self)
        self.imageQueue = imageQueue
        self.fileList = fileList
        self.imageDir = imageDir
        self.scaleFactor = scaleFactor

    def run(self):
        for name in self.fileList:
            try:
                name = self.imageDir + name
                imOrg = cv2.imread(name)
                if imOrg is None:
                    print(name, " ERROR - could not read image.", file=sys.stderr)
                    self.imageQueue.put(False)
                else:
                    if self.scaleFactor != 1.0:
                        imOrg = cv2.resize(imOrg, dsize=(0,0), fx=self.scaleFactor, fy=self.scaleFactor, interpolation=cv2.INTER_AREA)
                    self.imageQueue.put(imOrg)
            except cv2.error as e:
                print(name, " ERROR - cv2.error", str(e), file=sys.stderr)
                self.imageQueue.put(False)
            except:
                print(name, " ERROR - UNKNOWN:", sys.exc_info()[0], file=sys.stderr)
                self.imageQueue.put(False)

        self.imageQueue.put(None)

class TupleReaderThread(threading.Thread):
    def __init__(self, imageQueue, fileList, tupleSize=1):
        threading.Thread.__init__(self)
        self.imageQueue = imageQueue
        self.fileList = fileList
        self.tupleSize = tupleSize

    def run(self):
        for line in self.fileList:
            images = []
            for image_path in line.split():
                try:
                    img = cv2.imread(image_path,-1)
                    if img is None:
                        print(image_path, " ERROR - could not read image.", file=sys.stderr)
                        break
                    else:
                        images.append(img)
                except cv2.error as e:
                    print(image_path, " ERROR - cv2.error", str(e), file=sys.stderr)
                    break
                except:
                    print(image_path, " ERROR - UNKNOWN:", sys.exc_info()[0], file=sys.stderr)
                    break
            if len(images) == self.tupleSize:
                self.imageQueue.put(images)
            else:
                print(line, " ERROR - could not parse line.", file=sys.stderr)
                self.imageQueue.put(False)
        self.imageQueue.put(None)

class WriterThread(threading.Thread):
    def __init__(self, queue, path):
        threading.Thread.__init__(self)
        self.queue = queue
        self.path = path

    def run(self):
        while True:
            name, data = self.queue.get()
            if name is None:
                break

            if data.shape[2] == 1 or data.shape[2] == 3:
                name = os.path.basename(name)
                name += '.png'
                cv2.imwrite(os.path.join(self.path, name), data)
                #imgOut = cv2.resize(imgOut, dsize=(img.shape[1],img.shape[0]))
                #original[:,:,0] = np.repeat(np.mean(original, axis=2, keepdims=True), 3, axis=2)
                #original[:,:,0] *= 1-imgOut* 1.3
                #original[:,:,1] *= 1-imgOut* 1.3
                #original[:,:,2] *= imgOut* 1.3
                #cv2.imshow('OUT2', original /255)
                #cv2.waitKey(1)
                #cv2.imwrite('%s-shown.png' % fileName, original)
            else:
                name += '.npz'
                np.savez_compressed(os.path.join(self.path + name), data=data)


def generateDefocusPSF(radius):
    scale = 15.0
    psfRadius = int(radius * scale + 0.5)
    center = int((int(radius)+2)*scale+scale/2)
    psf = np.zeros((2*center,2*center))
    cv2.circle(psf, (center,center), psfRadius, color=1.0, thickness=-1, lineType=cv2.CV_AA if 2 == int(cv2.__version__.split(".")[0]) else cv2.LINE_AA)
    psf = cv2.resize(psf, dsize=(0,0), fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_AREA)
    psf = psf / np.sum(psf)
    return psf, int(center/scale)


def stackKernels(kernel0, kernel1):
    """Compute the cv2.filter2D of two kernels and returns the kernel of size
    [max(kernel0[0], kenrel1[0]), max(kernel0[1], kenrel1[1])] which is ceiled to nearest higher odd number

    Note
    ----
    Be aware of cv.2filter2D which makes correlation instead of convolution

    Parameters
    ----------
    kernel0 : numpy.ndarray
            2D kernel
    kernel1 : numpy.ndarray
            2D kernel

    Returns
    -------
    numpy.ndarray
            The computed cv2.filter2D kernel composed of kernel0 and kernel1 of size
            [max(kernel0[0], kenrel1[0]), max(kernel0[1], kenrel1[1])] which is ceiled to nearest higher odd number
    """
    x = kernel0.shape[0] if(kernel0.shape[0] >= kernel1.shape[0]) else kernel1.shape[0]
    y = kernel0.shape[1] if(kernel0.shape[1] >= kernel1.shape[1]) else kernel1.shape[1]
    x_odd = (x+1) if(x%2 == 0) else x
    y_odd= (y+1) if(y%2 == 0) else y
    v_center = np.asarray((x_odd/2, y_odd/2), dtype=np.int)
    kernel = np.zeros((x_odd,y_odd), dtype=np.float)
    kernel[v_center[0]][v_center[1]] = 1.0
    kernel = cv2.filter2D(kernel, -1, kernel0, borderType=cv2.BORDER_REPLICATE)
    kernel = cv2.filter2D(kernel, -1, kernel1, borderType=cv2.BORDER_CONSTANT)

    return kernel

def generateMotionBlurPSF(RNG, slope_deg=np.asarray([90,90]), length=np.asarray([0,9])):
    """Compute the motion blur PSF (Point Spread Function).

    Note
    -----
    The PSF has always the odd size.

    Parameters
    ----------
    RNG : numpy.random.RandomState() object
        Random state for sampling the slope_deg
    slope_deg : numpy.array
        the half open [low, high) interval of slope of the motion vector related to x axe in degrees to uniformly sample from
    length : numpy.array
        the half open [low, high) interval of motion vector length in pixels to uniformly sample from

    Returns
    -------
    numpy.ndarray
        The computed PSF kernel
    float
        sampled slope deg
    float
        sampled length"""
    supersample_coef = 100
    supersample_thickness = 100 / 10
    sampled_slope_deg = RNG.uniform(low=slope_deg[0], high=slope_deg [1])
    sampled_length = RNG.uniform(low=length[0], high=length [1])

    if(sampled_length == 0.0):
        return np.ones((1, 1), dtype=float)

    int_sampled_length = np.ceil(sampled_length).astype(np.int)
    kernel_size_odd = int_sampled_length + 1 if(int_sampled_length % 2 == 0) else int_sampled_length
    int_sup_sampled_length = np.rint(supersample_coef * sampled_length).astype(np.int)
    kernel_sup_size_odd = int(int_sup_sampled_length + 1 if (int_sup_sampled_length % 2 == 0) else int_sup_sampled_length)

    kernel_supersample = np.zeros([kernel_sup_size_odd, kernel_sup_size_odd], dtype=np.float)
    v_center_sup_kernel = (int(kernel_sup_size_odd / 2.), int(kernel_sup_size_odd / 2.))
    cv2.line(kernel_supersample, (0, v_center_sup_kernel[1]), (kernel_sup_size_odd - 1, v_center_sup_kernel[1]), color=(1), thickness=int(supersample_thickness * sampled_length))
    rot_mat = cv2.getRotationMatrix2D(center=v_center_sup_kernel, angle=sampled_slope_deg, scale=1)

    psf = cv2.warpAffine(src=kernel_supersample, M=rot_mat, dsize=kernel_supersample.shape)
    psf = cv2.resize(psf, dsize=(kernel_size_odd, kernel_size_odd), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    if(psf.shape != (1, 1)):
        psf = psf / psf.sum()
    return psf, sampled_slope_deg, sampled_length


def generateShakePSF(RNG, startSamples=500, length=300, halflife=0.5, resolution=15):
    superSampling = 5

    # generate random acceleration
    a = RNG.randn( 2, length+startSamples)
    # integrate speed
    a[0, :] = ewma(a[0, :], halflife=halflife * length)
    a[1, :] = ewma(a[1, :], halflife=halflife * length)
    # integrate position
    a = np.cumsum(a, axis=1)
    # skip first startSamples
    a = a[:, startSamples:]
    # center the kernel
    a = a - np.mean(a, axis=1).reshape((2,1))
    # normalize size
    maxDistance = ((a[0,:]**2 + a[1,:]**2) ** 0.5).max()
    a = a / maxDistance

    psf, t, t = np.histogram2d(a[0, :], a[1, :], bins=resolution * superSampling, range=[[-1.0, +1.0], [-1.0, +1.0]], normed=True)
    psf = cv2.resize(psf, (resolution, resolution), interpolation=cv2.INTER_AREA)
    psf = psf.astype(np.float32)
    psf = psf / np.sum(psf)
    return psf

def addArtefactsJPEG(img, quality):
    result, buffer = cv2.imencode('.jpg', img, [1, int(quality)])
    return cv2.imdecode(buffer, -1)

def gammaCorrection(img, gamma, maxVal=1):
    return (img**gamma) / (maxVal**gamma) * maxVal

def colorBalance(img, balance):
    newImg = np.copy(img)
    newImg[:, :, 0] *= balance[0]
    newImg[:, :, 1] *= balance[1]
    newImg[:, :, 2] *= balance[2]
    return newImg

def grayShiftScale(img, low, high):
    return img/255*(high-low) + low

def addAdditiveWhiteNoise(RNG, img, sdev):
    return img + RNG.randn(*img.shape) * sdev



def generateRandomCrop(RNG, images, patchSize, minEnergy=20, centerRegion=32):

    if type(images) is list:
        img = images[0]
    else:
        img = images

    centerRegion = min( [centerRegion, img.shape[0], img.shape[1]])
    centerPad = int((patchSize-centerRegion)/2)
    maxPos = [x-patchSize for x in img.shape[0:2]]

    energy = 0.0
    iteration = 0
    while energy < minEnergy and iteration < 10:
        p = [int(RNG.uniform(0, x)) for x in maxPos]
        cropImg = img[p[0]:p[0]+patchSize, p[1]:p[1]+patchSize, :]
        energy = np.std(cropImg[centerPad:centerPad+centerRegion,centerPad:centerPad+centerRegion, :])
        iteration += 1

    if type(images) is list:
        cropImg = [cropImg]
        for img in images[1:]:
            cropImg.append(img[p[0]:p[0]+patchSize, p[1]:p[1]+patchSize, :])

    return cropImg

def generateRandomCropCoord(RNG, image, patchSize, minEnergy = 5, energyAreaSize = -1):
    """Generate the crop coordinates [y,x,height,width].

    Note
    -----
    Coordinates are: [up_left_y, up_left_x, height, width]

    Parameters
    ----------
    RNG : numpy.random.RandomState() object
        Random state for sampling the crop coords
    image : np.ndarray
        the image in np.ndarray format
    patch_size : int
        width

    Returns
    -------
    numpy.ndarray
        Coordinates are as [up_left_y, up_left_x, height, width]
    double
        energy (np.std) of the energyAreaSize defined patch
    """
    max_y = image.shape[0]-patchSize
    max_x = image.shape[1]-patchSize
    if(max_y < 0 or max_x < 0):
        return None
    if energyAreaSize < 0:
        energyAreaSize = patchSize

    energyArea = np.asarray([int((patchSize-energyAreaSize)/2.), int((patchSize-energyAreaSize)/2.), energyAreaSize, energyAreaSize], dtype = np.uint)

    patchCoord = None
    energy, loopCnt = 0, 0

    while energy < minEnergy and loopCnt < 10 :
        sample_y = RNG.randint(0, max_y)
        sample_x = RNG.randint(0, max_x)
        patchCoord = np.asarray([sample_y, sample_x, patchSize, patchSize], dtype=np.uint)
        patch = image[patchCoord[0]:patchCoord[0]+patchCoord[2], patchCoord[1]:patchCoord[1]+patchCoord[3]]
        energy = np.std(patch[energyArea[0]:energyArea[0]+energyArea[2], energyArea[0]:energyArea[0]+energyArea[2]])
#         print ("CR Energe: %f" % (energy))
#         cv2.imshow("patch", patch)
#         cv2.imshow("energy", patch[energyArea[0]:energyArea[0]+energyArea[2], energyArea[0]:energyArea[0]+energyArea[2]])
#         cv2.waitKey()
        loopCnt += 1

    return patchCoord, energy #np.asarray((sample_y, sample_x, patchSize, patchSize), dtype=np.uint)


def generateRandomCrop2(RNG, img, labels, patchSize):
    maxPos = [x-patchSize for x in img.shape[0:2]]
    p = [int(RNG.uniform(0, x)) for x in maxPos]
    cropImg = img[p[0]:p[0]+patchSize, p[1]:p[1]+patchSize, :]
    cropLabel = labels[p[0]:p[0]+patchSize, p[1]:p[1]+patchSize, :]

    return cropImg, cropLabel


def warpPerspective(rx, ry, rz, fov, img, positions=None, shift=(0,0)):

    s = max(img.shape[0:2])
    rotVec = np.asarray((rx*np.pi/180,ry*np.pi/180, rz*np.pi/180))
    rotMat, j = cv2.Rodrigues(rotVec)
    rotMat[0, 2] = 0
    rotMat[1, 2] = 0
    rotMat[2, 2] = 1

    f = 0.3
    trnMat1 = np.asarray(
        (1, 0, -img.shape[1]/2,
         0, 1, -img.shape[0]/2,
         0, 0, 1)).reshape(3, 3)

    T1 = np.dot(rotMat, trnMat1)
    distance = (s/2)/math.tan(fov*np.pi/180)
    T1[2, 2] += distance


    cameraT = np.asarray(
        (distance, 0, img.shape[1]/2 + shift[1],
         0, distance, img.shape[0]/2 + shift[0],
         0, 0, 1)).reshape(3,3)

    T2 = np.dot(cameraT, T1)

    newImage = cv2.warpPerspective(img, T2, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LANCZOS4)
    if positions is None:
        return newImage
    else:
        return newImage, np.squeeze( cv2.perspectiveTransform(positions[None, :, :], T2), axis=0)
