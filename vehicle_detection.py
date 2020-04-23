import sys
import numpy as np
import cv2


"""
    A program for detecting military vehicles in nature photos
    
    Built using OpenCV 4.1.0 and NumPy 1.16.1
    
    Utilizing OpenCV library for Hough transformation, morphology, K-means clustering and Canny edge detection
    
    Arguments:
    1 - input image
    2 - kernel size
    3 - low threshold for Canny
    4 - high threshold for Canny
    5 - number of clusters for K-means clustering   (set to 0 to ignore operation)
    6 - size of SE mask                             (set to 0 to ignore operation)
    7 - rho value for Hough transform
    8 - threshold value for Hough transform
    9 - maximum line gap for Hough transform
    10 - minimum line length for Hough transform
    
    Example usage:
    python vd.py kampvogner2.jpg 9 75 150 0 3 2 150 10 30
    
"""


# Method for smoothening the image using Gaussian blur
def noise_reduction(image, kernel_size):
    smooth = cv2.GaussianBlur(
        image,
        kernel_size,
        0
    )

    return smooth


# Method for edge detection using Canny edge detection
def edge_detection(image, threshold_low, threshold_high):
    edges = cv2.Canny(
        image,
        threshold1=threshold_low,
        threshold2=threshold_high,
        L2gradient=True
    )

    return edges


# Method for morphological dilation
def dilation(image, size):
    dilated = cv2.dilate(
        image,
        cv2.getStructuringElement(cv2.MORPH_DILATE, (size, size))
    )

    return dilated


# Method for K-means clustering
def color_quantization(image, K):
    Z = image.reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(
        np.float32(Z),
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    center = np.uint8(center)
    res = center[label.flatten()]
    segm = res.reshape((image.shape))

    return segm


# Method for line linking and highlighting, using the Hough transformation
def line_detection(image, rho, threshold, maxLineGap, minLineLength):

    # Copy grayscale image to RGB image in order to display detected lines and intersects in red color
    detected_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detected_intersects = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Detect and store lines in input image
    lines = cv2.HoughLinesP(
        image,
        rho=rho,
        theta=np.pi / 180,
        threshold=threshold,
        maxLineGap=maxLineGap,
        minLineLength=minLineLength
    )

    # Loop through all lines and draw them in the image
    if lines is not None:

        # Prepare storage for all detected intersects
        intersects = []

        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(detected_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)

            # Nested loop for getting intersect between all lines
            for j in range(0, len(lines)):
                k = lines[j][0]

                if (l is not None and k is not None) and (k is not l):
                    point = line_intersection(l, k)

                    if point[0] is not None and point[1] is not None:
                        x = int(round(point[0]))
                        y = int(round(point[1]))
                        intersects.append((x, y))
                        # Comment the following line out if you dont wish to mark intersects on lines-image output
                        cv2.ellipse(detected_lines, (x, y), (5, 5), 0, 0, 360, (0, 255, 0), 1)

        # Plot intersects
        if intersects is not None:
            for k in range(0, len(intersects)):
                x = int(round(intersects[k][0]))
                y = int(round(intersects[k][1]))
                cv2.ellipse(detected_intersects, (x, y), (5, 5), 0, 0, 360, (255, 255, 0), 1)

    return detected_lines, detected_intersects


# Method for detecting intersecting lines
def line_intersection(line1, line2):

    # Line 1 coordinates
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    # Line 2 coordinates
    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    denominator = ((x1 - x2) * (y3 - y4) - ((y1 - y2) * (x3 - x4)))

    # Declare intersect coordinates
    p_x = None
    p_y = None

    # If denominator is zero, the lines are parallel or coincide
    if denominator is not 0:

        t = (((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))) / denominator

        u = (((x1 - x2) * (y1 - y3)) - ((y1 - y2) * (x1 - x3))) / denominator

        # Calculate the intersecting points
        if 0.0 <= t <= 1.0:
            p_x = (x1 + (t * (x2 - x1)))
            p_y = (y1 + (t * (y2 - y1)))
        elif 0.0 <= u <= 1.0:
            p_x = (x3 + (u * (x4 - x3)))
            p_y = (y3 + (u * (y4 - y3)))

    return (p_x, p_y)


# Method for detecting BLOBs
def blob_detection(image):

    # Set parameters for the BLOB detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 75
    params.maxThreshold = 125
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 10000
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)

    detected_blobs = detector.detect(image)

    # Creates and returns image with BLOB plots using the built-in drawKeyPoints() method
    #marked_image = cv2.drawKeypoints(
    #    image,
    #    detected_blobs,
    #    np.array([]),
    #    (255, 255, 0),
    #    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    #)

    #return marked_image

    return detected_blobs


# Method for plotting BLOBs
def plot_blobs(image):
    plotted_blobs = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    blobs = blob_detection(intersects)

    for blob in blobs:
        x = int(blob.pt[0])
        y = int(blob.pt[1])
        w = int(blob.size)
        cv2.rectangle(plotted_blobs, (x - w, y - w), (x + w, y + w), (255, 255, 0), 1)

    return plotted_blobs


# Read input image and kernel sizes
input_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
kernel_size = (int(sys.argv[2]), int(sys.argv[2]))

# Check if kernel size is an even number number
if int(sys.argv[2]) % 2 == 0:
    print("Kernel: Size must be an odd number")
    exit(1)

# Resize image to fit screen
image_resized = cv2.resize(input_image, (1024, 768))

# Smooth the input image
smooth = noise_reduction(image_resized, kernel_size)
cv2.imshow('Smoothened', smooth)

# Read parameters for the edge detection operation
threshold_low = int(sys.argv[3])
threshold_high = int(sys.argv[4])

# Check if threshold values ratio are in the range of 2:1 to 3:1, in accordance with Canny edge detection best practice
if threshold_high < (threshold_low * 2) or threshold_high > (threshold_low * 3):
    print("Canny edge detection: threshold high and low ratio should be in the range of 2:1 to 3:1")
    exit(1)

# Read K value for color quantization
clusters = int(sys.argv[5])

# If value is 0, ignore segmentation operation and jump straight to edge detection
if clusters == 0:
    edges = edge_detection(smooth, threshold_low, threshold_high)

else:
    # Check if value of K for K Means Clustering is between legal range
    if not 1 <= clusters <= 255:
        print("K Means Clustering: K must be between 1 and 255")
        exit(1)
    else:
        # Perform segmentation
        segm = color_quantization(cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR), clusters)
        cv2.imshow('Segmented', segm)
        # Perform edge detection
        edges = edge_detection(segm, threshold_low, threshold_high)

cv2.imshow('Edges', edges)

# Read value for SE mask size
SE_mask = int(sys.argv[6])

# Read parameters for the edge linking operation
rho = int(sys.argv[7])
threshold = int(sys.argv[8])
maxLineGap = int(sys.argv[9])
minLineLength = int(sys.argv[10])

# If value is 0, ignore dilation operation and jump straight to line linking
if SE_mask == 0:
    lines, intersects = line_detection(edges, rho, threshold, maxLineGap, minLineLength)
else:
    # Check if value of SE mask size is of odd number
    if SE_mask % 2 == 0 or SE_mask < 0:
        print("Dilation: SE mask size must be a positive and odd number")
        exit(1)
    else:
        # Perform dilation
        dilated = dilation(edges, SE_mask)
        cv2.imshow('Dilated', dilated)
        # Perform Hough transform to detect lines
        lines, intersects = line_detection(dilated, rho, threshold, maxLineGap, minLineLength)

cv2.imshow('Lines', lines)
cv2.imshow('Intersects', intersects)

# Detect and plot BLOBs
plotted_blobs = plot_blobs(image_resized)
cv2.imshow('BLOBs', plotted_blobs)

cv2.waitKey(0)
