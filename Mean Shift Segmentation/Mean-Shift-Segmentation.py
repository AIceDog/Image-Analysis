###########################################################################################
#
# In this project, the algorithm consists of 3 steps for each iteration which are
# mean shift filtering, merging similar clusters and row-by-row merging similar pixels.
# for .ppm image, we use feature vector coordinate + color
# for .pgm image, we use feature vector coordinate + grayscale
#
###########################################################################################

import cv2, random, sys, re, math, time, numpy as np
from scipy.spatial import distance
from numpy import *

# choose image to process
def Choose_Image():
    _image_Origin, _isPgm, _isPpm = [], False, False

    while not (type(_image_Origin) is np.ndarray):
        input_ = input("\nPlease enter the picture name:\n"
                       "\tSupported format：.pgm   .ppm:"
                       "\n:")

        if re.findall(".pgm", input_):
            _isPgm, _image_Origin = True, cv2.imread(input_, cv2.COLOR_BGR2GRAY)
        elif re.findall(".ppm", input_):
            _isPpm, _image_Origin = True, cv2.imread(input_, 1)
        elif re.findall("end", input_):
            print("Program terminated.")
            sys.exit(0)

        if not (type(_image_Origin) is np.ndarray):
            print("Image doesn't exist, please try again.\nOr enter 'end' to terminate the program.\t")
            _isPgm, _isPpm = False, False

    return _image_Origin, _isPgm, _isPpm

# set bandwidth of color or grayscale and bandwidth of space
def Set_Bandwidth(_isPgm, _isPpm):
    _bandWidth_COG, _bandWidth_S = 0, 0

    while (True):
        if _isPgm == True:
            input_ = input("\nPlease set bandwidth for grayscale:"
                            "\n:")
        elif _isPpm == True:
            input_ = input("\nPlease set bandwidth for color:"
                            "\n:")

        if input_.isdigit() and int(input_) > 0:
            _bandWidth_COG = int(input_)
            break
        elif input_.isdigit() and int(input_) == 0:
            print("bandwidth cannot be 0.")
        elif input_ == 'end':
            print("Program terminated.")
            sys.exit(0)
        else:
            print("Input is invalid, please try again.\nOr enter 'end' to terminate the program.\t")

    while (True):
        input_ = input("\nPlease set bandwidth for space:"
                        "\n:")

        if input_.isdigit() and int(input_) > 0:
            _bandWidth_S = int(input_)
            break
        elif input_.isdigit() and int(input_) == 0:
            print("bandwidth cannot be 0.")
        elif input_ == 'end':
            print("Program terminated.")
            sys.exit(0)
        else:
            print("Input is invalid, please try again.\nOr enter 'end' to terminate the program.\t")

    return _bandWidth_COG, _bandWidth_S

# computer new cluster center by gaussian kernel function
def Compute_New_Cluster_Center(_pixels_FV, _cluCen_FV_Prev, _isPgm, _isPpm):  # _pixels_FV 里面装的是要计算的像素的特征向量
    _pixels_S = np.array(_pixels_FV)[:, 0:2] # get a list of positions to be computed
    _pixels_COG = np.array(_pixels_FV)[:, 2:] # get a list of color or grayscale values to be computed

    _cluCen_Prev_S = np.array(_cluCen_FV_Prev[0:2]) # position of center pixel
    _cluCen_Prev_COG = np.array(_cluCen_FV_Prev[2:]) # color or grayscale of center pixel

    # compute denominator
    _denominator_Arr = np.exp(-0.5 * (np.sum((_pixels_S - _cluCen_Prev_S) ** 2, axis=1) / bandWidth_S ** 2 +
                                      np.sum((_pixels_COG - _cluCen_Prev_COG) ** 2, axis=1) / bandWidth_COG ** 2))
    _denominator = sum(_denominator_Arr)

    # compute numerator
    if _isPgm == True:
        _numerator = sum(np.array(_pixels_FV) * tile(_denominator_Arr.reshape(len(_denominator_Arr), 1), (1, 3)),
                         axis=0)
    elif _isPpm == True:
        _numerator = sum(np.array(_pixels_FV) * tile(_denominator_Arr.reshape(len(_denominator_Arr), 1), (1, 5)), axis=0)

    _new_CluCen = _numerator / _denominator
    return _new_CluCen

# functions of grouping pixels with similar color or grayscale through row-by-row searching(like connected components)
def Find(_x, _parent):
    _i = _x
    while 0 != _parent[_i]:
        _i = _parent[_i]
    return _i
def Union(_firstNode, _secondNode, _parent):
    _second, _first = _secondNode, _firstNode

    while 0 != _parent[_first]:
        _first = _parent[_first]
    while 0 != _parent[_second]:
        _second = _parent[_second]

    if _first < _second:
        _parent[_second] = _first
    if _first > _second:
        _parent[_first] = _second

    return _parent
def Two_Pass(_rowNum, _colNum, _parent, _labels, _image_ToBeProcess, _image_Label, _thre_COG_MSC_RowByRow):  # 参数：行，列
    # first pass
    _label, _left, _up = 0, 0, 0

    for row in range(0, _rowNum):
        for column in range(0, _colNum):
            if column > 0 and (abs(_image_ToBeProcess[row][column] - _image_ToBeProcess[row][column - 1]) <= _thre_COG_MSC_RowByRow + 1).all():
                _left = _image_Label[row][column - 1]
            else:
                _left = 0

            if row > 0 and (abs(_image_ToBeProcess[row][column] - _image_ToBeProcess[row - 1][column]) <= _thre_COG_MSC_RowByRow + 1).all():
                _up = _image_Label[row - 1][column]
            else:
                _up = 0

            if _left != 0 or _up != 0:
                if _left != 0 and _up != 0:
                    _image_Label[row][column] = min(_left, _up)
                    if _left != _up:
                        _parent = Union(_up, _left, _parent)
                else:
                    _image_Label[row][column] = max(_left, _up)

            else:
                _label = _label + 1
                _image_Label[row][column] = _label

    # second pass
    for row in range(0, _rowNum):
        for column in range(0, _colNum):
            _image_Label[row][column] = Find(_image_Label[row][column], _parent)
            _labels.append(_image_Label[row][column])

    _labels = np.unique(_labels)
    return _labels, _parent, _image_Label


img_Ori = [] # original image
isPgm = False # whether we choose .pgm image
isPpm = False # whether we choose .pgm image

bandWidth_COG = 0   # bandwidth for color or grayscale
bandWidth_S = 0     # bandwidth for space

img_Ori, isPgm, isPpm = Choose_Image()
bandWidth_COG, bandWidth_S = Set_Bandwidth(isPgm, isPpm)

rowNum, colNum = img_Ori.shape[0:2]

# threshold of color or grayscale in the process of merging similar clusters and row-by-row searching
thre_COG_MSC_RowByRow = bandWidth_COG


cv2.imshow('original image', img_Ori)
cv2.waitKey(0)


start = time.perf_counter()
########## region mean shift filtering
print("processing meanshift filtering")

# create indices of rectangle with length of side ((fil_HalfSize * 2) + 1)
fil_HalfSize = 8 # the half size of filter
fil_Indices = [] # indices of filter
for row in range(-fil_HalfSize, fil_HalfSize + 1):
    for column in range(-fil_HalfSize, fil_HalfSize + 1):
        fil_Indices.append([row, column])
fil_Indices = np.array(fil_Indices)

if isPgm == True:
    pix_FV_Ori = np.ones((rowNum, colNum, 3))  # feature vector (row, column, grayscale)
    pix_FV_Ori[:, :, 2] = img_Ori
elif isPpm == True:
    pix_FV_Ori = np.ones((rowNum, colNum, 5))  # feature vector (row, column, red, green, blue)
    pix_FV_Ori[:, :, 2:] = img_Ori
for row in range(0, rowNum):
    for column in range(0, colNum):
        pix_FV_Ori[row][column][0], pix_FV_Ori[row][column][1] = row, column

# pix_Ignored stores marks which indicate whether pixels have been processed
# if pixels have been processed and can be ignored, the value of element in pix_Ignored will be 1
pix_Ignored = np.zeros((rowNum, colNum))
cluCen_FV_AF = []     # stores feature vectors of each cluster center after meanshift filtering
pixPos_EvClu_AF = [[]]  # stores positions of pixels in every cluster after meanshift filtering
threshold_Filter = 4    # meanshift filter to decide whether or not to continue clustering

# process each pixel of image
for row in range(0, rowNum):
    for column in range(0, colNum):
        if pix_Ignored[row, column] == 1: # if the pixel has been processed before
            continue

        pixPos_EvClu_AF.append([])
        pix_Ignored[row, column] = 1

        _dist = 100000  # distance between precious cluster center and current cluster center
        cluCen_FV_Curr = []  # feature vector of current cluster center
        cluCen_FV_Prev = pix_FV_Ori[row, column]    # feature vector of previous cluster center
        _row, _column = row, column

        # if distance between precious cluster center and current cluster center is greater than threshold_Filter
        # continue while loop
        while _dist > threshold_Filter:
            # store a list of feature vectors of pixels which will be processed in Compute_New_Cluster_Center()
            pix_FV_ToBeProc = []

            # delete the indices which are out of image range(like[-1, 0], [-1, -1])
            fil_InImg_Indices = fil_Indices[(fil_Indices[:, 0] + _row >= 0) &
                                            (fil_Indices[:, 0] + _row < rowNum) &
                                            (fil_Indices[:, 1] + _column >= 0) &
                                            (fil_Indices[:, 1] + _column < colNum)]
            fil_InImg_Indices[:, 0] += _row
            fil_InImg_Indices[:, 1] += _column

            # meanShift_Indices stores the indices of pixels which will be computed in Compute_New_Cluster_Center()
            meanShift_Indices = [pos for pos in fil_InImg_Indices
                                 if abs(pos[0] - _row) < bandWidth_S and \
                                 abs(pos[1] - _column) < bandWidth_S and \
                                 sum((pix_FV_Ori[pos[0], pos[1], 2:] - cluCen_FV_Prev[2:]) ** 2) < bandWidth_COG ** 2]

            # opt_Indices stores the indices of pixels which will be optimized
            # that is, those pixels will be ignored next time they occur in for loop, even they are not truly computed
            opt_Indices = [pos for pos in meanShift_Indices
                           if abs(pos[0] - _row) < 6 and \
                           abs(pos[1] - _column) < 6]

            # pix_FV_ToBeProc stores feature vectors of pixels which will be computed in Compute_New_Cluster_Center()
            for pos in meanShift_Indices:
                pix_FV_ToBeProc.append(pix_FV_Ori[pos[0], pos[1]])

            for pos in opt_Indices:
                pix_Ignored[pos[0], pos[1]] = 1  # mark the pixel processed
                pixPos_EvClu_AF[-1].append([pos[0], pos[1]])

            if len(pix_FV_ToBeProc) > 0:
                cluCen_FV_Curr = Compute_New_Cluster_Center(pix_FV_ToBeProc, cluCen_FV_Prev, isPgm, isPpm)
                # compute the distance between previous cluster center and current cluster center
                _dist = np.sqrt(sum((np.array(cluCen_FV_Curr[2:]) - np.array(cluCen_FV_Prev[2:])) ** 2))
                cluCen_FV_Prev = cluCen_FV_Curr
                _row, _column = round(cluCen_FV_Prev[0]), round(cluCen_FV_Prev[1])
            else:
                _dist = 0

        cluCen_FV_AF.append(cluCen_FV_Prev)

del(pixPos_EvClu_AF[0])
########## endregion
end = time.perf_counter()
cost = end - start
print("processing time", cost, "s")
print()
########## region show image
if isPgm == True:
    img_Filtered = np.ones((rowNum, colNum, 1), np.float)
elif isPpm == True:
    img_Filtered = np.ones((rowNum, colNum, 3), np.float)

if isPgm == True:
    for i in range(0, len(cluCen_FV_AF)):
        for pos in pixPos_EvClu_AF[i]:
            img_Filtered[pos[0], pos[1]] = cluCen_FV_AF[i][2]
elif isPpm == True:
    for i in range(0, len(cluCen_FV_AF)):
        for pos in pixPos_EvClu_AF[i]:
            img_Filtered[pos[0], pos[1]] = cluCen_FV_AF[i][2:]

cv2.imshow('meanshift filtering', img_Filtered.astype(np.uint8))
cv2.waitKey(0)
########## endregion



start = time.perf_counter()
########## region merge similar clusters
print("merging similar clusters")

# cluCen_FV_MSC stores the feature vectors of each cluster center in the process of Merging Similar Clusters
cluCen_FV_MSC = cluCen_FV_AF.copy()

if isPgm == True:
    img_MSC = np.ones((rowNum, colNum, 1), np.float)
elif isPpm == True:
    img_MSC = np.ones((rowNum, colNum, 3), np.float)

# pixPos_EvClu_AF stores positions of pixels in every cluster after meanshift filtering
while len(pixPos_EvClu_AF) > 1:
    indices_ToBeProc = [0]  # the indices of clusters to be merged
    pix_Num = len(pixPos_EvClu_AF[0])   # pix_Num stores the total number of pixels in several similar clusters

    for i in range(1, len(cluCen_FV_MSC)):
        # if color or grayscale distance between two cluster > thre_COG_MSC_RowByRow
        if (abs(cluCen_FV_MSC[0][2:] - cluCen_FV_MSC[i][2:]) > thre_COG_MSC_RowByRow).any():
            continue

        # we want the clusters whose color or grayscale distance from cluCen_FV_MSC[0]
        # are less than thre_COG_MSC_RowByRow
        indices_ToBeProc.append(i)
        pix_Num += len(pixPos_EvClu_AF[i])

        # if shortest space distance between two cluster(pixPos_EvClu_AF[0] and pixPos_EvClu_AF[i]) <= 2
        if min(distance.cdist(np.array(pixPos_EvClu_AF[0]), np.array(pixPos_EvClu_AF[i])).min(axis=1)) <= 2:
            indices_ToBeProc.append(i)
            pix_Num += len(pixPos_EvClu_AF[i])

    ### region compute weight average mean of color or grayscale in several similar clusters
    if len(indices_ToBeProc) > 1:  # if cluster pixPos_EvClu_AF[0] finds similar clusters
        if isPgm == True:
            cog_WeiMean = 0.
        elif isPpm == True:
            cog_WeiMean = np.array([0.0, 0.0, 0.0])

        weiArr = np.array([len(pixPos_EvClu_AF[i]) for i in indices_ToBeProc]) / pix_Num

        if isPgm == True:
             weiMat = tile(weiArr.reshape(len(weiArr), 1), (1, 1)) # weight matrix for weight mean computing
             COG_Array = np.array([cluCen_FV_MSC[i][2:] for i in indices_ToBeProc])
             cog_WeiMean = (COG_Array * weiMat).sum()  # compute weight average mean of color
        elif isPpm == True:
            weiMat = tile(weiArr.reshape(len(weiArr), 1), (1, 3))  # weight matrix for weight mean computing
            COG_Array = np.array([cluCen_FV_MSC[i][2:] for i in indices_ToBeProc])
            cog_WeiMean = (COG_Array * weiMat).sum(0)  # compute weight average mean of color

        # place all position information of pixels in other similar clusters into the first cluster pixPos_EvClu_AF[0]
        for i in indices_ToBeProc:
            pixPos_EvClu_AF[0] += pixPos_EvClu_AF[i]
        for pos in pixPos_EvClu_AF[0]:
            if isPgm == True:
                img_MSC[pos[0], pos[1]] = cog_WeiMean
            elif isPpm == True:
                img_MSC[pos[0], pos[1]] = list(cog_WeiMean)
    ### endregion

    elif len(indices_ToBeProc) == 1: # if cluster pixPos_EvClu_AF[0] doesn't find similar clusters
        for pos in pixPos_EvClu_AF[0]:
            if isPgm == True:
                img_MSC[pos[0], pos[1]] = list(cluCen_FV_MSC[0][2])
            elif isPpm == True:
                img_MSC[pos[0], pos[1]] = list(cluCen_FV_MSC[0][2:])

    # delete computed clusters and find other new similar clusters
    cluCen_FV_MSC = np.delete(cluCen_FV_MSC, indices_ToBeProc, axis=0)
    pixPos_EvClu_AF = [pixPos_EvClu_AF[i] for i in range(len(pixPos_EvClu_AF)) if
                        i not in indices_ToBeProc]

img_MSC = img_MSC.astype(np.uint8)
########## endregion
end = time.perf_counter()
cost = end - start
print("processing time", cost, "s")
print()
######### region show image
cv2.imshow('merge similar clusters', img_MSC)
cv2.waitKey(0)
######### endregion



start = time.perf_counter()
########## region row-by-row merging similar pixels and eliminating small clusters
print("row-by-row merging similar pixels and eliminating small clusters")

labels = []
parent = [0] * (rowNum * colNum)
img_Label = np.zeros((rowNum, colNum), dtype = np.int)
img_Seg = img_MSC.copy()

######### step 1 row by row merging similar pixels and regenerate clusters
labels, parent, img_Label = Two_Pass(rowNum, colNum, parent, labels, img_Seg, img_Label, thre_COG_MSC_RowByRow)


######### step 2 eliminate small clusters
largeClu_Pos = []   # largeClu_Pos stores the positions of pixels in every large cluster
smallClu_Pos = []   # smallClu_Pos stores the positions of pixels in every small cluster

#### region create a list of small clusters and a list of large clusters
for i in labels:
    indices = np.where(img_Label == i)
    one_cluster_Pos = list(zip(indices[0], indices[1]))
    if len(list(one_cluster_Pos)) <= 30: # clusters of which pixel number is less than 30
        smallClu_Pos.append(list(one_cluster_Pos)) # add pixel position information of small clusters into smallClu_Pos
    elif len(list(one_cluster_Pos)) > 30:
        largeClu_Pos.append(list(one_cluster_Pos)) # add pixel position information of small clusters into smallClu_Pos
    one_cluster_Pos.clear()

# img_MarkSmallClu mark which pixels are in small clusters
# if one pixel is in a small cluster, then its corresponding coordinate in img_MarkSmallClu is 1
img_MarkSmallClu = np.zeros((rowNum, colNum), dtype = np.int)
for oneClu in smallClu_Pos:
    for pos in oneClu:
        img_MarkSmallClu[pos[0], pos[1]] = 1
#### endregion

#### region compute average color in large area
# store average color or grayscale of each cluster
if isPgm == True:
    clu_COG_Aver = np.zeros((len(largeClu_Pos), 1), np.float)
elif isPpm == True:
    clu_COG_Aver = np.zeros((len(largeClu_Pos), 3), np.float)

# compute average color
for i in range(0, len(largeClu_Pos)):
    for pos in largeClu_Pos[i]:
        clu_COG_Aver[i] += img_Seg[pos[0], pos[1]]
    clu_COG_Aver[i] /= len(largeClu_Pos[i])

for i in range(0, len(largeClu_Pos)):
    for pos in largeClu_Pos[i]:
        img_Seg[pos[0], pos[1]] = clu_COG_Aver[i]
#### endregion

#### region merge small area
for oneClu in smallClu_Pos:
    # get the pixel coordinates around each small cluster(four direction, up, down, left, right)
    periPixels = list(np.array(oneClu) + [-1, 0]) + \
                 list(np.array(oneClu) + [1, 0]) + \
                 list(np.array(oneClu) + [0, -1]) + \
                 list(np.array(oneClu) + [0, 1])
    # delete repetitive coordinates
    periPixels = np.unique(np.array(periPixels), axis = 0)
    # delete coordinate which is out of boundary of image
    periPixels = periPixels[(periPixels[:, 0] >= 0) &
                            (periPixels[:, 0] < rowNum) &
                            (periPixels[:, 1] >= 0) &
                            (periPixels[:, 1] < colNum)]
    periPixels = [x for x in periPixels if img_MarkSmallClu[x[0], x[1]] == 0]

    # if the peripheral pixels are in big clusters, then assign color to the small cluster
    if len(periPixels) > 0:
        for pos in oneClu:
            img_Seg[pos[0], pos[1]] = img_Seg[periPixels[0][0], periPixels[0][1]]
#### endregion
########## endregion
end = time.perf_counter()
cost = end - start
print("processing time", cost, "s")
######### region show image
#cv2.imshow('image after row by row merging', img_Seg.astype(np.uint8))
cv2.imshow('eliminate small clusters', img_Seg.astype(np.uint8))
cv2.waitKey(0)
######### endregion
