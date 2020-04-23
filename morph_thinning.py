import skimage
import scipy
import sys
import numpy as np
from skimage import util, io
from copy import deepcopy


# Hit-and-miss method
# takes a subimage of the input image and one SE mask
# and matches them against eachother
def ham(subimage, se):
    # Iterate over the subimage received and check if there is a match against the SE
    # If it's a negative SE value, ignore
    for i in range(subimage.shape[0]):
        for j in range(subimage.shape[1]):
            if se[i, j] == -1:
                continue
            elif not subimage[i, j] == se[i, j]:
                return False

    return True


# Read input image
image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=np.ubyte)

# Expected output image
expected_out = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]], dtype=np.ubyte)
# Set SE masks
se_masks = []
se_masks.append(np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]]))
se_masks.append(np.array([[-1, 0, 0], [1, 1, 0], [1, 1, -1]]))
se_masks.append(np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]]))
se_masks.append(np.array([[1, 1, -1], [1, 1, 0], [-1, 0, 0]]))
se_masks.append(np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]]))
se_masks.append(np.array([[-1, 1, 1], [0, 1, 1], [0, 0, -1]]))
se_masks.append(np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]]))
se_masks.append(np.array([[0, 0, -1], [0, 1, 1], [-1, 1, -1]]))

# Set padding size (assuming all SE masks are equally shaped)
pad_h, pad_w = se_masks[0].shape[0]//2, se_masks[0].shape[1]//2

# Pad the output image
padded = util.pad(image, (pad_h, pad_w), mode="constant")

# Prepare output image that copies all contents from the padded image
out = deepcopy(padded)

# Run loop
while True:
    # Break-condition variable
    equal = True

    # Iterate over the SE masks
    for l in range(len(se_masks)):
        # Copy contents of the output image into the padded image
        # used for referencing
        padded = deepcopy(out)

        # Iterate over the padded image in a row-wise fashion
        for i in range(1, padded.shape[0]):
            for j in range(1, padded.shape[1]):
                # Extract subimage form the padded image
                subim = padded[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]

                # Check if the subimage matches the subimage exactly
                # and "delete" (set to background) the current element if so
                # Will leave the break-condition variable as True if the method
                # returns True (match between SE and subimage)
                if ham(subim, se_masks[l]):
                    out[i, j] = 0
                    equal = False
    if equal:
        break

# Crop and print the output image
cropped = out[pad_h:image.shape[0]+pad_h, pad_w:image.shape[1]+pad_w]
print(cropped)

print(expected_out)
