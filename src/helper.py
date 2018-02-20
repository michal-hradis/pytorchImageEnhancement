
import numpy as np
import cv2

def generateMotionBlurPSF(slope_deg, length):
    """Create a linear  motion blur PSF (Point Spread Function).

        Note
        -----
        The PSF image has always the odd size.

        Parameters
        ----------
        slope_deg : motion direction in degrees
        length : motion length in pixels

        Returns
        -------
        numpy.ndarray
            The computed PSF kernel
        """
    supersample_coef = 5
    supersample_thickness = supersample_coef

    if(length == 0.0):
        return np.ones((1, 1), dtype=float)

    int_length = np.ceil(length).astype(np.int)
    kernel_size_odd = (int_length / 2) * 2 + 1
    sup_kernel_size = np.rint(
        supersample_coef * kernel_size_odd).astype(np.int)

    sup_kernel = np.zeros(
        [sup_kernel_size, sup_kernel_size], dtype=np.float)

    sup_radius = (length * supersample_coef) / 2

    slope = slope_deg / 180 * np.pi
    pos = np.array([sup_radius, 0])
    R = np.array([
        [np.cos(-slope), -np.sin(-slope)],
        [np.sin(-slope), np.cos(-slope)]])
    pos = R.dot(pos)
    center = np.array(
        [float(sup_kernel_size) / 2,
         float(sup_kernel_size) / 2])

    cv2.line(
        sup_kernel,
        tuple(np.rint(center - pos).astype(np.int32)),
        tuple(np.rint(center + pos).astype(np.int32)),
        color=(1), thickness=supersample_thickness)

    psf = cv2.resize(
        sup_kernel, dsize=(int(kernel_size_odd), int(kernel_size_odd)),
        interpolation=cv2.INTER_AREA)

    #cv2.imshow('large', sup_kernel)
    psf = psf / psf.max()
    #cv2.imshow('small', psf)
    #cv2.waitKey()
    psf = psf / psf.sum()
    return psf


def collage(data, normSamples=False):
    images = [img for img in data.transpose(0, 2, 3, 1)]
    if normSamples:
        for img in images:
            img += img.min()
            img /= img.max()

    side = int(np.ceil(len(images)**0.5))
    for i in range(side**2 - len(images)):
        images.append(images[-1])
    collage = [np.concatenate(images[i::side], axis=0)
               for i in range(side)]
    collage = np.concatenate(collage, axis=1)
    #collage -= collage.min()
    #collage = collage / np.absolute(collage).max() * 256
    return collage