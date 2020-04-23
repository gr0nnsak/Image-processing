import skimage
import scipy
import sys
import numpy as np
from skimage import util,io,color
from scipy import ndimage

"""

	Manual implementation of the Canny Edge detector

	Built using Scikit Image, SciPy and Numpy

"""

# Read in image and convert to appropriate format
image = util.img_as_float(color.rgb2gray(io.imread(sys.argv[1])))

# Get image dimensions
height, width = image.shape

# Read in sigma
sigma = float(sys.argv[2])

# Calculate size of filter mask
size = int(6*sigma)-1

# Create the two sobel filter masks
sobel_x = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


# Function for creating Gaussian 2D filter mask (ref. page 730)
# G(x,y) = e^-(x^2+y^2)/(2*sigma^2)
def gaussian2d(size, sigma):
    mask = np.ndarray((size, size), dtype=np.float)

    for i in range(size):
        for j in range(size):
            x, y = dist(size, i, j)
            mask[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

    return mask


# Function for converting top-left aligned coordinates to center aligned coordinates
def dist(size, i, j):
    center = size//2

    return i-center, j-center


# Function for getting the gradient magnitude (ref. pages 716, 717-720)
def gradient(image):

    grad_x = ndimage.convolve(image, sobel_x, mode="nearest")
    grad_y = ndimage.convolve(image, sobel_y, mode="nearest")

    grad = (grad_x ** 2 + grad_y ** 2) ** 0.5

    angle = np.arctan2(grad_y, grad_x)

    return grad, angle


# Function for calculating nonmax suppression
def nonmax(gradient, angle):
    nonmax = np.zeros_like(gradient, dtype=np.int32)

    q = 255
    r = 255

    for i in range(1, height-1):
        for j in range(1, width-1):

            # 0 degrees
            if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                q = gradient[i, j + 1]
                r = gradient[i, j - 1]
            # 45 degrees
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient[i + 1, j - 1]
                r = gradient[i - 1, j + 1]
            # 90 degrees
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient[i + 1, j]
                r = gradient[i - 1, j]
            # 135 degrees
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient[i - 1, j - 1]
                r = gradient[i + 1, j + 1]

            if image[i, j] >= q and image[i, j] >= r:
                nonmax[i, j] = gradient[i, j]
            else:
                nonmax[i, j] = 0

    return nonmax


# Function for double thresholding (ref. page 759)
def hysteresis_treshold(nonmax):

    # Threshold limits
    t_h = np.amax(nonmax) * 0.15
    t_l = t_h * 0.05

    # Set intensity values for weak and strong pixels
    weak = 75
    strong = 255

    # Resulting threshold image
    thresh = np.zeros_like(nonmax, dtype=np.int32)

    # Threshold images, strong and weak, respectively
    high = np.zeros_like(nonmax, dtype=np.int32)
    low = np.zeros_like(nonmax, dtype=np.int32)

    # Determine values of high- and low-value threshold images
    for i in range(1, height-1):
        for j in range(1, width-1):
            if nonmax[i, j] >= t_h:
                high[i, j] = strong
            if t_h > nonmax[i, j] >= t_l:
                low[i, j] = weak

    # Merge high and low into threshold image
    for k in range(height):
        for l in range(width):
            if not low[k, l] == 0:
                thresh[k, l] = low[k, l]
            if not high[k, l] == 0:
                thresh[k, l] = high[k, l]

    # Perform hysteresis thresholding
    for x in range(height):
        for y in range(width):
            if thresh[i, j] == weak:
                if nonmax[i+1, j-1] == 255 or nonmax[i+1, j] == 255 or nonmax[i+1, j+1] == 255 or nonmax[i, j-1] == 255 or nonmax[i, j+1] == 255 or nonmax[i-1, j-1] == 255 or nonmax[i-1, j] == 255 or nonmax[i-1, j+1] == 255:
                    thresh[i, j] = 255
                else:
                    thresh[i, j] = 0

    return thresh


# Smooth the image using Gaussian 2D
filt = gaussian2d(size, sigma)
gauss_smooth = ndimage.convolve(image, filt, mode="nearest")

# Get the gradient and angle of the smoothened image
gradient, angle = gradient(gauss_smooth)

# Perform nonmax suppression
nonmax = nonmax(gradient, angle)

# Perform hysteresis thresholding
thresh = hysteresis_treshold(nonmax)

# Save the output image
io.imsave("out.png", util.img_as_ubyte(thresh))
