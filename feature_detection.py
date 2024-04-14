import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
from itertools import product

import transformations

## Helper functions ############################################################

def inbounds(shape, indices):
    '''
        Input:
            shape -- int tuple containing the shape of the array
            indices -- int list containing the indices we are trying 
                       to access within the array
        Output:
            True/False, depending on whether the indices are within the bounds of 
            the array with the given shape
    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Compute Harris Values ############################################################

def computeHarrisValues(srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'. Also compute an
        # orientation for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        i_xp = scipy.ndimage.sobel(srcImage, axis=1, mode='nearest')
        i_yp = scipy.ndimage.sobel(srcImage, axis=0, mode='nearest')

        w_p_00 = scipy.ndimage.gaussian_filter(np.square(i_xp), sigma=0.5, truncate=3.0, mode='nearest')
        w_p_01 = scipy.ndimage.gaussian_filter(np.multiply(i_xp, i_yp), sigma=0.5, truncate=3.0, mode='nearest')
        w_p_10 = w_p_01
        w_p_11 = scipy.ndimage.gaussian_filter(np.square(i_yp), sigma=0.5, truncate=3.0, mode='nearest')

        harris_matrix = np.zeros((2, 2))
        for y in range(height):
            for x in range(width):
                # w_yx = w_p[y, x]
                harris_matrix[0, 0] = w_p_00[y, x]
                harris_matrix[0, 1] = w_p_01[y, x]
                harris_matrix[1, 0] = w_p_10[y, x]
                harris_matrix[1, 1] = w_p_11[y, x]

                harrisImage[y, x] = np.linalg.det(harris_matrix) - 0.1 * (np.trace(harris_matrix) ** 2)

        orientationImage = np.degrees(np.arctan2(i_yp, i_xp))
        # TODO-BLOCK-END

        return harrisImage, orientationImage


## Compute Corners From Harris Values ############################################################


def computeLocalMaximaHelper(harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, dtype=bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        height, width = harrisImage.shape
        max_filter = scipy.ndimage.filters.maximum_filter(harrisImage, size=(7,7), mode='nearest')
        for y in range(height):
            for x in range(width):
                destImage[y, x] = True if harrisImage[y, x] == max_filter[y, x] else False
        # TODO-BLOCK-END

        return destImage


def detectCorners(harrisImage, orientationImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        Output:
            features -- list of all detected features. Entries should
            take the following form:
            (x-coord, y-coord, angle of gradient, the detector response)

            x-coord: x coordinate in the image
            y-coord: y coordinate in the image
            angle of the gradient: angle of the gradient in degrees
            the detector response: the Harris score of the Harris detector at this point
        '''
        height, width = harrisImage.shape[:2]
        features = []

        # TODO 3: Select the strongest keypoints in a 7 x 7 area, according to
        # the corner strength function. Once local maxima are identified then
        # construct the corresponding corner tuple of each local maxima.
        # Return features, a list of all such features.
        # TODO-BLOCK-BEGIN
        destImage = computeLocalMaximaHelper(harrisImage)
        for y in range(height):
             for x in range(width):
                  if destImage[y, x]:
                       corner_tuple = (x, y, orientationImage[y, x], harrisImage[y,x])
                       features.append(corner_tuple)
        # TODO-BLOCK-END

        return features


## Compute MOPS Descriptors ############################################################
def computeMOPSDescriptors(image, features):
    """"
    Input:
        image -- Grayscale input image in a numpy array with
                values in [0, 1]. The dimensions are (rows, cols).
        features -- the detected features, we have to compute the feature
                    descriptors at the specified coordinates
    Output:
        desc -- K x W^2 numpy array, where K is the number of features
                and W is the window size
    """
    image = image.astype(np.float32)
    image /= 255.
    # This image represents the window around the feature you need to
    # compute to store as the feature descriptor (row-major)
    windowSize = 8
    desc = np.zeros((len(features), windowSize * windowSize))
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)

    for i, f in enumerate(features):
        transMx = np.zeros((2, 3))

        # TODO 4: Compute the transform as described by the feature
        # location/orientation and store in 'transMx.' You will need
        # to compute the transform from each pixel in the 40x40 rotated
        # window surrounding the feature to the appropriate pixels in
        # the 8x8 feature descriptor image. 'transformations.py' has
        # helper functions that might be useful
        # Note: use grayImage to compute features on, not the input image
        # TODO-BLOCK-BEGIN

        r_m = transformations.get_rot_mx(0, 0, -1*f[2] * np.pi / 180)
        t1_m_init= transformations.get_trans_mx(np.array([-f[0], -f[1], 0]))
        t2_m_init = transformations.get_trans_mx(np.array([4, 4, 0]))
        s_m = transformations.get_scale_mx(0.2, 0.2, 1)

        transMx_init = t2_m_init @ s_m @ r_m @ t1_m_init
        transMx = transMx_init[:2, :]
        transMx = np.delete(transMx, 2, axis=1)

  
        # TODO-BLOCK-END

        # Call the warp affine function to do the mapping
        # It expects a 2x3 matrix
        destImage = cv2.warpAffine(grayImage, transMx,
            (windowSize, windowSize), flags=cv2.INTER_LINEAR)

        # TODO 5: Normalize the descriptor to have zero mean and unit
        # variance. If the variance is negligibly small (which we
        # define as less than 1e-10) then set the descriptor
        # vector to zero. Lastly, write the vector to desc.
        # TODO-BLOCK-BEGIN

        sub_mean = np.array(destImage) - np.mean(np.array(destImage))
        if np.var(np.array(destImage)) < float('1e-10'):
            normalized = 0
        else:
            normalized = sub_mean / np.std(np.array(destImage))
            normalized = normalized.flatten()

        desc[i] = normalized
        # TODO-BLOCK-END

    return desc


## Compute Matches ############################################################
def produceMatches(desc_img1, desc_img2):
    """
    Input:
        desc_img1 -- corresponding set of MOPS descriptors for image 1
        desc_img2 -- corresponding set of MOPS descriptors for image 2

    Output:
        matches -- list of all matches. Entries should 
        take the following form:
        (index_img1, index_img2, score)

        index_img1: the index in corners_img1 and desc_img1 that is being matched
        index_img2: the index in corners_img2 and desc_img2 that is being matched
        score: the scalar difference between the points as defined
                    via the ratio test
    """
    matches = []
    assert desc_img1.ndim == 2
    assert desc_img2.ndim == 2
    assert desc_img1.shape[1] == desc_img2.shape[1]

    if desc_img1.shape[0] == 0 or desc_img2.shape[0] == 0:
        return []

    # TODO 6: Perform ratio feature matching.
    # This uses the ratio of the SSD distance of the two best matches
    # and matches a feature in the first image with the closest feature in the
    # second image. If the SSD distance is negligibly small, in this case less 
    # than 1e-5, then set the distance to 1. If there are less than two features,
    # set the distance to 0.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # TODO-BLOCK-BEGIN

    # Handling when there's less than 2 features but more than 0
    f_length = desc_img1.shape[1]
    if desc_img1.shape[0] == 1:
        for i in range(desc_img2.shape[0]):
            matches.append((0, i, 0))
        return matches
    if desc_img2.shape[0] == 1:
        for i in range(desc_img1.shape[0]):
            matches.append((i, 0, 0))
        return matches

    for i, feature in enumerate(desc_img1):
        dist_list = list(map(lambda x: scipy.spatial.distance.cdist(np.reshape(feature, (1, f_length)), np.reshape(x, (1,f_length)))[0, 0] ** 2, desc_img2))
        print(f"dist list is {dist_list}")
        distances = np.array(dist_list)
        print(f"distances is {distances}")

        closest = np.argmin(distances)
        closest_dist = distances[closest]
        if closest_dist < float('1e-5'):
            matches.append((i, closest, 1))
            continue

        distances = np.delete(distances, closest)
        second_dist = distances.min(distances)

        ratio = closest_dist / second_dist

        matches.append((i, closest, ratio))
    # TODO-BLOCK-END

    return matches




