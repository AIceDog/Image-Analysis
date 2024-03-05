###########################################################################################
#
#   This project implements image segmentation with K-means algorithm.
#
#   You can select .ppm image and .pgm image
#   You can set the number of clusters in the image
#   You can set threshold for K-means to stop the iteration
#
###########################################################################################

import cv2, random, sys, re
import numpy as np


image_Origin = [] # original image
image_Result = [] # result image to be shown
name = "" # image name

isPgm = False # whether we choose .pgm image
isPpm = False # whether we choose .pgm image

# the feature vector mode to be calculated
# FV_Mode == 0  --->  pgm : grayscale, ppm : color
# FV_Mode == 1  --->  pgm : location, ppm : location
# FV_Mode == 2  --->  pgm : grayscale + location, ppm : color + location
FV_Mode = 0

cluster_Number = 0 # the number of clusters
variance = 99999 # variance between previous feature vector centers and current feature vector centers
threshold = 0 # threshold controls when to stop iteration


########## region Choose image
while not (type(image_Origin) is np.ndarray):
    input_ = input("\nPlease enter the picture name:\n"
                   "\tSupported format：.pgm   .ppm:"
                   "\n:")

    if re.findall(".pgm", input_):
        name = input_
        isPgm = True
        image_Origin = cv2.imread(name, cv2.COLOR_BGR2GRAY)
    elif re.findall(".ppm", input_):
        name = input_
        isPpm = True
        image_Origin = cv2.imread(name, 1)
    elif re.findall("end", input_):
        print("Program terminated.")
        sys.exit(0)

    if not (type(image_Origin) is np.ndarray):
        print("Image doesn't exist, please try again.\nOr enter 'end' to terminate the program.\t")
        isPgm = False
        isPpm = False
########## endregion


########## region Choose image type
if isPpm == True:
    while (True):
        input_ = input("\nPlease set the image type:\n"
                       "\tInput 'color' or 'grayscale':"
                       "\n:")
        if input_ == 'color':
            break
        elif input_ == 'grayscale':
            image_Origin = cv2.cvtColor(image_Origin, cv2.COLOR_BGR2GRAY)
            isPpm = False
            isPgm = True
            break
        elif input_ == 'end':
            print("Program terminated.")
            sys.exit(0)
        else:
            print("Input is invalid, please try again.\nOr enter 'end' to terminate the program.\t")
########## endregion


########## region Choose feature vector mode
while (True):
    if isPgm == True:
        input_ = input("\nPlease select feature vector mode:\n"
                       "\tpress 'a' : grayscale,\n\tpress 'b' : location,\n\tpress 'c' : grayscale + location,"
                       "\n:")

    elif isPpm == True:
        input_ = input("\nPlease select feature vector mode:\n"
                       "\tpress 'a' : color,\n\tpress 'b' : location,\n\tpress 'c' : color + location,"
                       "\n:")

    if input_ == 'a':
        FV_Mode = 0
        break
    elif input_ == 'b':
        FV_Mode = 1
        break
    elif input_ == 'c':
        FV_Mode = 2
        break
    elif input_ == 'end':
        print("Program terminated.")
        sys.exit(0)
    else:
        print("Input is invalid, please try again.\nOr enter 'end' to terminate the program.\t")
########## endregion


########## region Set number of clusters
while (True):
    input_ = input("\nPlease set number of clusters:\n"
                   "\n:")

    if input_.isdigit() and int(input_) > 0:
        cluster_Number = int(input_)
        break
    elif input_.isdigit() and int(input_) == 0:
        print("Number of clusters cannot be 0.")
    elif name == 'end':
        print("Program terminated.")
        sys.exit(0)
    else:
        print("Input is invalid, please try again.\nOr enter 'end' to terminate the program.\t")
########## endregion


########## region Set threshold
while (True):
    input_ = input("\nPlease set threshold:\n"
                   "\n:")

    if input_.isdigit() and int(input_) > 0:
        threshold = int(input_)
        break
    elif input_.isdigit() and int(input_) == 0:
        print("Threshold cannot be 0.")
    elif name == 'end':
        print("Program terminated.")
        sys.exit(0)
    else:
        print("Input is invalid, please try again.\nOr enter 'end' to terminate the program.\t")
########## endregion



image_Result = image_Origin.copy()
row_Number, column_Number = image_Origin.shape[0:2]

# image_Label records the label of each pixel, that is which cluster does each pixel belong to
image_Label = np.zeros((row_Number, column_Number))
image_Label[:] = -1

# image_Dist record the nearest distance of each pixel to the clusters
image_Dist = np.zeros((row_Number, column_Number))
image_Dist[:] = -1

# feature_Vectors store the whole feature vector
# for .ppm image, it is color + location
# for .pgm image, it is grayscale + location
if isPgm == True:
    feature_Vectors = np.ones((row_Number, column_Number, 3))
    feature_Vectors[:, :, 0] = image_Origin
elif isPpm == True:
    feature_Vectors = np.ones((row_Number, column_Number, 5))
    feature_Vectors[:, :, :-2] = image_Origin

for row in range(0, row_Number):
    for column in range(0, column_Number):
        feature_Vectors[row][column][-2] = row
        feature_Vectors[row][column][-1] = column


ori_Cluster_Centers = [] # store original feature vector centers
cur_Cluster_Centers = [] # store current feature vector centers
prev_Cluster_Centers = [] # store previous feature vector centers

if isPgm == True:
    cur_Cluster_Centers = np.zeros((cluster_Number, 3), dtype=np.float)
elif isPpm == True:
    cur_Cluster_Centers = np.zeros((cluster_Number, 5), dtype=np.float)


########## region Choose initial cluster centers
# choose initial cluster centers based on kmeans++ algorithm
# firstly randomly choose a pixel
cur_Cluster_Centers[0][-2] = random.randint(0, row_Number)
cur_Cluster_Centers[0][-1] = random.randint(0, column_Number)
cur_Cluster_Centers[0][:-2] = image_Origin[int(cur_Cluster_Centers[0][-2])][int(cur_Cluster_Centers[0][-1])]

ori_Cluster_Centers.append(cur_Cluster_Centers[0])


# secondly choose remaining cluster centers based on distance between each pixel and chosen cluster centers
for i in range(0, cluster_Number - 1):

    for row in range(0, row_Number):
        for column in range(0, column_Number):

            if FV_Mode == 0:   # ppm : color, pgm : grayscale
                dist = np.sum(np.square(feature_Vectors[row, column, :-2] - np.array(ori_Cluster_Centers)[:, :-2]), axis=1)

            elif FV_Mode == 1:   # ppm : location, pgm : location
                dist = np.sum(np.square(feature_Vectors[row, column, -2:] - np.array(ori_Cluster_Centers)[:, -2:]), axis=1)

            elif FV_Mode == 2:   # ppm : color + location, pgm : grayscale + location
                dist = np.sum(np.square(feature_Vectors[row, column] - np.array(ori_Cluster_Centers)[:]), axis=1)

            image_Dist[row, column] = np.min(dist, axis=0)

    # find the pixel whose distance is furthest from chosen cluster centers
    next_Cluster_Index = np.unravel_index(image_Dist.argmax(), image_Dist.shape)

    cur_Cluster_Centers[i + 1][-2] = next_Cluster_Index[0]
    cur_Cluster_Centers[i + 1][-1] = next_Cluster_Index[1]
    cur_Cluster_Centers[i + 1][:-2] = image_Origin[int(cur_Cluster_Centers[i + 1][-2])][int(cur_Cluster_Centers[i + 1][-1])]

    ori_Cluster_Centers.append(cur_Cluster_Centers[i + 1])
########## endregion



########## region kmeans algorithm
# if variance between previous cluster centers and current cluster centers
# is smaller than threshold, then terminate the while loop
while (variance >= threshold):

    prev_Cluster_Centers = cur_Cluster_Centers

    # calculate distance between each pixel and each cluster center
    # reassign each pixel to the closest cluster center
    for row in range(0, row_Number):
        for column in range(0, column_Number):

            if FV_Mode == 0:   # ppm : color, pgm : grayscale
                dist = np.sum(np.square(feature_Vectors[row, column, :-2] - cur_Cluster_Centers[:, :-2]), axis=1)

            elif FV_Mode == 1:   # ppm : location, pgm : location
                dist = np.sum(np.square(feature_Vectors[row, column, -2:] - cur_Cluster_Centers[:, -2:]), axis=1)

            elif FV_Mode == 2:   # ppm : color + location, pgm : grayscale + location
                dist = np.sum(np.square(feature_Vectors[row, column] - cur_Cluster_Centers[:]), axis=1)

            image_Label[row, column] = np.argmin(dist, axis=0)

    # recalculate cluster centers and get new cluster centers
    for number in range(0, cluster_Number):
        _indices = np.where(np.isin(image_Label[:], number))
        length = len(_indices[0])
        cur_Cluster_Centers[number] = feature_Vectors[_indices[0], _indices[1]].sum(axis=0) / length

    # computer variance between previous cluster centers and current cluster centers
    variance = np.sum(np.square(cur_Cluster_Centers - prev_Cluster_Centers))
########## endregion



# assign cluster mean values to corresponding pixels
for number in range(0, cluster_Number):
    _indices = np.where(np.isin(image_Label[:], number))
    indices = list(zip(_indices[0], _indices[1]))

    if isPgm == True:
        for index in indices:
            image_Result[index[0]][index[1]] = cur_Cluster_Centers[number][0]
    elif isPpm == True:
        for index in indices:
            image_Result[index[0]][index[1]] = cur_Cluster_Centers[number][:-2]


cv2.imshow('image_result', image_Result)
cv2.waitKey(0)
