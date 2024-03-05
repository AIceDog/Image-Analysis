###########################################################################################
#
#   This is a program of connected components labeling of row by row method.
#
#   You can choose three types of image format which are .pgm, .pbm and .ppm
#
#   At the same time, this program computes features of each connected component including area, perimeter,
#   centroid, circularity1, circularity2, bounding box, second moments, angle of axis of least interia
#
###########################################################################################
import cv2
import math
import random
import numpy as np
import re
import sys
import pandas as pd
import plotly.graph_objects as go


def start_function():
    print("--------------------------Thanks for using this program!----------------------\n"
          "- The program is to find objects in the images.                              -\n"
          "-     And extract their main features in order to determine similar objects. -\n"
          "--------------------------------Let's get started!----------------------------\n")

# Input the name of the picture
def input_function():
    name = input("Please enter the picture name:\n"
                 "\tSupported format：.pgm   .pbm   .ppm"
                 "\n:")
    if re.findall(".pgm", name) or re.findall(".pbm", name) or re.findall(".ppm", name):
        return name
    elif re.findall("END", name):
        print("Program terminated.")
        sys.exit(0)
    else:
        print("Invalid enter, please try again.\n\tOr enter 'END' to terminate the program.\t")
        name = input_function()
        return name


# Read image and determine if the picture exists.
def image_read_function():
    name = input_function()
    image_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    # Two types of values may be encountered, None and ndarray.
    # Convert them to the same value format for better judgment.
    temp_value = np.array(image_gray)

    while not temp_value.any():
        print("Image doesn't exist, please try again.\n\tOr enter 'END' to terminate the program.\t")
        name = input_function()
        image_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        temp_value = np.array(image_gray)

    # if image size is too large, we reduce it
    if image_gray.shape[0] * image_gray.shape[1] > 2073600:
        image_gray = cv2.imread(name, cv2.IMREAD_REDUCED_GRAYSCALE_2)

    return name, image_gray


# Union-Find Find function, find the root node
def Find(x, parent):
    i = x

    while 0 != parent[i]:
        i = parent[i]

    return i


# Union-Find Union function, union two node
def Union(firstNode, secondNode, parent):
    second = secondNode
    first = firstNode

    while 0 != parent[first]:
        first = parent[first]
    while 0 != parent[second]:
        second = parent[second]

    if first < second:
        parent[second] = first
    if first > second:
        parent[first] = second

    return parent


# two pass function
def Two_Pass(MaxRow, MaxCol, parent, labels):  # 参数：行，列
    global image_binary
    global image_label

    # first pass
    label = 0

    for row in range(0, MaxRow):
        for column in range(0, MaxCol):
            if image_binary[row][column] != 0:

                left = image_label[row][column - 1] if column > 0 else 0
                up = image_label[row - 1][column] if row > 0 else 0

                if left != 0 or up != 0:
                    if left != 0 and up != 0:
                        image_label[row][column] = min(left, up)
                        if left != up:
                            parent = Union(up, left, parent)
                    else:
                        image_label[row][column] = max(left, up)
                else:
                    label = label + 1
                    image_label[row][column] = label


    # second pass
    for row in range(0, MaxRow):
        for column in range(0, MaxCol):
            if image_binary[row][column] != 0:
                image_label[row][column] = Find(image_label[row][column], parent)
                labels.append(image_label[row][column])

    labels = np.unique(labels)
    return labels, parent


# generate bounding box of each connected component
def Generate_BoundingBox(binary_img, regions):
    for oneRegion in regions:
        oneRegionT = list(map(list, zip( * oneRegion)))

        # up most
        minRow = min(oneRegionT[0])

        # down most
        maxRow = max(oneRegionT[0])

        # left most
        minCol = min(oneRegionT[1])

        # right most
        maxCol = max(oneRegionT[1])
        cv2.rectangle(binary_img, (minCol, minRow), (maxCol, maxRow), (1, 1, 1), 1)


# get a list of peripheral pixels of a connected component
def Get_Peripheral_Pixels(region):
    peripheral_pixels = []

    # to test which pixels are in the periphery
    periphery_detect = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # up right down left

    # get an unordered peripheral pixel list, and store them in periphery_disorder
    for pixel in region:
        is_peripheral_pixel = True
        for detect in periphery_detect:
            is_peripheral_pixel = is_peripheral_pixel & ([pixel[0] + detect[0], pixel[1] + detect[1]] in region)
        if is_peripheral_pixel == False:
            peripheral_pixels.append(pixel)

    return peripheral_pixels


# compute perimeter of each connected component
def Compute_Perimeter(region):
    # used to get ordered peripheral pixels
    #   [-1, -1]    [-1, 0]    [-1, 1]
    #   [ 0, -1] current pixel [ 0, 1]
    #   [ 1, -1]    [ 1, 0]    [ 1, 1]
    # the detect direction is clockwise and current pixel is in the center
    detect_ring = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    # store peripheral pixels but pixel arrange is disordered
    periphery_disorder = Get_Peripheral_Pixels(region)

    # store peripheral pixels and pixel arrange is ordered
    periphery_order = []

    perimeter = 0

    # to get an ordered peripheral pixel list, we compute pixel by pixel
    # current_pixel means the current pixel which is the center of the clockwise searching
    current_pixel = min(periphery_disorder) # we choose the top pixel at the begining
    periphery_order.append(current_pixel)

    # start_index is the start index of detect_ring, used to detect the next pixel
    start_index = 0

    for i in detect_ring:
        if ((current_pixel[0] + i[0], current_pixel[1] + i[1]) in periphery_disorder):
            # if we find the next pixel, we mark the new start position of searching next periphery pixel
            start_index = detect_ring.index(i) + 5 if detect_ring.index(i) + 5 < 8 else detect_ring.index(i) - 3

            current_pixel = [current_pixel[0] + i[0], current_pixel[1] + i[1]]
            periphery_order.append(current_pixel)
            break

    while current_pixel != list(periphery_order[0]):
        currentIndex = start_index
        for i in range(0, 8):

            currentIndex = currentIndex + 1 if currentIndex + 1 < 8 else 0

            if ((current_pixel[0] + detect_ring[currentIndex][0],
                 current_pixel[1] + detect_ring[currentIndex][1]) in periphery_disorder):
                start_index = currentIndex + 5 if currentIndex + 5 < 8 else currentIndex - 3
                current_pixel = [current_pixel[0] + detect_ring[currentIndex][0],
                                 current_pixel[1] + detect_ring[currentIndex][1]]

                periphery_order.append(current_pixel)
                break

    # compute perimeter
    for i in range(0, len(periphery_order) - 1):
        perimeter += math.sqrt(pow(periphery_order[i][0] - periphery_order[i + 1][0], 2) + pow(
            periphery_order[i][1] - periphery_order[i + 1][1], 2))

    return perimeter


# compute area of each connected component
def Compute_Area(region):
    return len(region)


# compute centroid coordinate of each connected component
def Compute_Centroid(region):
    centroid_row = 0
    for pixel in region:
        centroid_row = centroid_row + pixel[0]
    centroid_row = round(centroid_row / len(region), 2)

    centroid_col = 0
    for pixel in region:
        centroid_col = centroid_col + pixel[1]
    centroid_col = round(centroid_col / len(region), 2)

    return centroid_row, centroid_col


# compute circularity1 of each connected component
def Compute_Circularity1(perimeter, area):
    circularity1 = math.pow(perimeter, 2) / area
    return circularity1


# compute circularity2 of each connected component
def Compute_Circularity2(region):

    peripheral_pixels = Get_Peripheral_Pixels(region)

    centroid_row, centroid_col = Compute_Centroid(region)

    miR = 0

    for pixel in peripheral_pixels:
        miR = miR + math.sqrt(math.pow(pixel[0] - centroid_row, 2) + math.pow(pixel[1] - centroid_col, 2))

    miR = miR / len(peripheral_pixels)

    sigmaPow = 0

    for pixel in peripheral_pixels:
        sigmaPow = sigmaPow + math.pow(
            math.sqrt(
                math.pow(pixel[0] - centroid_row, 2) +
                math.pow(pixel[1] - centroid_col, 2)
            ) - miR,
            2)

    sigmaPow = sigmaPow / len(peripheral_pixels)

    circularity2 = miR / math.sqrt(sigmaPow)

    return circularity2


# compute second moments of each connected component
def Compute_Second_Moments(region):
    centroid_row, centroid_col = Compute_Centroid(region)
    area = Compute_Area(region)

    # second-order row moment
    muRR = 0

    for pixel in region:
        muRR = muRR + math.pow(pixel[0] - centroid_row, 2)

    muRR = muRR / area

    # second-order mixed moment
    muRC = 0

    for pixel in region:
        muRC = muRC + (pixel[0] - centroid_row) * (pixel[1] - centroid_col)

    muRC = muRC / area

    # second-order column moment
    muCC = 0

    for pixel in region:
        muCC = muCC + math.pow(pixel[1] - centroid_col, 2)

    muCC = muCC / area

    return muRR, muRC, muCC


# compute the orientation of least and most inertia axis of each connected component
def Compute_Least_Interia(muRR, muRC, muCC):
    tan2alpha = 2 * muRC / (muRR - muCC)
    alpha2 = math.atan(tan2alpha)
    alpha = alpha2 / 2
    return alpha


# generate random color to label each connected component
def Generate_Random_Color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return [red, green, blue]


# compute distance between two feature vectors
def Compute_Vector_Distance(vector1, vector2, weights):
    distance = 0
    for i in range(0, len(vector1)):
        distance += math.pow(vector1[i] - vector2[i], 2) * weights[i]
    distance = round(math.sqrt(distance), 2)
    return distance


if __name__ == "__main__":
    labels = []
    MaxRow = 0
    MaxCol = 0
    image_size = 0
    parent = [0] * 10000

    # connected_components is a 3 dimensional list which restore pixel coordinates of each connected components
    connected_components = []

    centroid_list = []      # store centroid coordinate of each connected component
    area_list = []          # store area of each connected component
    perimeter_list = []     # store perimeter of each connected component
    circularity1_list = []  # store circularity1 of each connected component
    circularity2_list = []  # store circularity2 of each connected component
    muRR_list = []          # store second-order row moment of each connected component
    muRC_list = []          # store second-order mixed moment of each connected component
    muCC_list = []          # store second-order column moment of each connected component
    angle_list = []         # store angle of axis of least inertia of each connected component
    color_list = []         # store colors to label each connected components

    start_function()

    image_name, image_gray = image_read_function()

    MaxRow, MaxCol = image_gray.shape
    image_size = MaxRow * MaxCol


############ test image format and perform thresholding and cleaning using morphological filters ############
    if image_name[-4:] == ".pgm" or image_name[-4:] == ".ppm":
        (thresh, image_binary) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    elif image_name[-4:] == ".pbm":
        image_gray = cv2.blur(image_gray, (4, 4))
        (thresh, image_binary) = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    if image_size < 480000:
        ksize = (1, 1)
    else:
        ksize = (2, 2)

    # use morphological filters to process image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
############ test image format and perform thresholding and cleaning using morphological filters ############


    # create large enough image_label to record label of each pixel
    image_label = np.array([[0] * MaxCol] * MaxRow)

    # use two pass function to implement connected components labeling
    labels, parent = Two_Pass(MaxRow, MaxCol, parent, labels)

    # get coordinates of all pixels of each connected components and restore them in connected_components
    for i in labels:
        indices = np.where(image_label == i)
        one_connected_component = list(zip(indices[0], indices[1]))
        connected_components.append(list(one_connected_component))
        one_connected_component.clear()


############ filter some connected components based on their properties ############
    # remove connected component of which pixel number is less than 20
    connected_components = [i for i in connected_components if len(i) > 20]

    # remove connected component which is originally background
    connected_components = [i for i in connected_components if len(i) < (image_size / 3) ]

    for one_connected_component in connected_components:
        centroid_list.append(Compute_Centroid(one_connected_component))

        area_list.append(Compute_Area(one_connected_component))

        perimeter_list.append(round(Compute_Perimeter(one_connected_component), 2))

        circularity1_list.append(round(Compute_Circularity1(perimeter_list[-1], area_list[-1]), 2))
        circularity2_list.append(round(Compute_Circularity2(one_connected_component), 2))

        muRR, muRC, muCC = Compute_Second_Moments(one_connected_component)
        muRR_list.append(round(muRR, 2))
        muRC_list.append(round(muRC, 2))
        muCC_list.append(round(muCC, 2))

        angle_list.append(round(Compute_Least_Interia(muRR, muRC, muCC), 2))

    # list index_to_delete records the index of connected components to be deleted
    index_to_delete = []
    for i in range(0, len(circularity1_list)):
        if circularity1_list[i] > 90:
            index_to_delete.append(i)

    # delete some connected components based on index_to_delete
    connected_components = [connected_components[i] for i in range(0, len(connected_components), 1) if
                            i not in index_to_delete]

    centroid_list = [centroid_list[i] for i in range(0, len(centroid_list), 1) if
                            i not in index_to_delete]
    area_list = [area_list[i] for i in range(0, len(area_list), 1) if
                            i not in index_to_delete]

    perimeter_list = [perimeter_list[i] for i in range(0, len(perimeter_list), 1) if
                            i not in index_to_delete]

    circularity1_list = [circularity1_list[i] for i in range(0, len(circularity1_list), 1) if
                            i not in index_to_delete]
    circularity2_list = [circularity2_list[i] for i in range(0, len(circularity2_list), 1) if
                            i not in index_to_delete]

    muRR_list = [muRR_list[i] for i in range(0, len(muRR_list), 1) if
                 i not in index_to_delete]
    muRC_list = [muRC_list[i] for i in range(0, len(muRC_list), 1) if
                 i not in index_to_delete]
    muCC_list = [muCC_list[i] for i in range(0, len(muCC_list), 1) if
                 i not in index_to_delete]

    angle_list = [angle_list[i] for i in range(0, len(angle_list), 1) if
                 i not in index_to_delete]
############ filter some connected components based on their properties ############


############ show processed images ############
    ##### show image after connected components labeling
    image_result = np.zeros((MaxRow, MaxCol))

    for i in range(0, len(connected_components)):
        for j in connected_components[i]:
            image_result[j[0]][j[1]] = 1

    print("number of connected components", len(connected_components))

    cv2.imshow('image_result', image_result)
    cv2.waitKey(0)


    ##### show image with bounding box
    Generate_BoundingBox(image_result, connected_components)

    cv2.imshow('image bounding box', image_result)
    cv2.waitKey(0)


    ##### generate color for each connected components
    for i in range(0, len(connected_components)):
        color_list.append(Generate_Random_Color())

    # image_color display each connected component with rgb color
    image_color = np.zeros((MaxRow, MaxCol), dtype = np.uint8)
    image_color = cv2.cvtColor(image_color, cv2.COLOR_GRAY2BGR)

    for i in range(0, len(connected_components)):
        for j in connected_components[i]:
            image_color[j[0]][j[1]] = color_list[i]

    ##### use data visualization to report features of each connected components
    colors = []
    for i in color_list:
        colors.append('rgb(' + str(i[0]) + ', ' + str(i[1]) + ', ' + str(i[2]) + ')')

    numbers = range(0, len(connected_components))

    data = {'Number' : numbers,
            'Color': color_list, 'Centroid': centroid_list, 'Area': area_list, 'Perimeter': perimeter_list,
            'Circularity1': circularity1_list, 'Circularity2': circularity2_list,
            'Second_Order_Row_Moment': muRR_list,
            'Second_Order_Mixed_Moment': muRC_list,
            'Second_Order_Column_Moment': muCC_list,
            'Orientation_Of_Least_And_Most_Inertia_Axis': angle_list}

    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Number</b>",
                    "<b>Color</b>", "<b>Centroid</b>", "<b>Area</b>", "<b>Perimeter</b>",
                    "<b>Circularity1</b>", "<b>Circularity2</b>",
                    "<b>muRR</b>", "<b>muRC</b>",
                    "<b>muCC</b>",
                    "<b>Least And Most Inertia Axis</b>"],
            line_color='black', fill_color='white',
            align='center', font=dict(color='black', size=15), height=30
        ),
        cells=dict(
            values=[df.Number,
                    df.Color, df.Centroid, df.Area, df.Perimeter,
                    df.Circularity1, df.Circularity2,
                    df.Second_Order_Row_Moment, df.Second_Order_Mixed_Moment, df.Second_Order_Column_Moment,
                    df.Orientation_Of_Least_And_Most_Inertia_Axis],
            line_color='black', fill_color=['white', colors, 'white'],
            align='center', font=dict(color='black', size=15), height=30)
        ),
    ])

    fig.show()



    ##### indicate which obejcts are similar based on area, perimeter, circularity1, circularity2, muRC
    # if distance between two feature vectors are less than the threshold, then the two connected components are silimiar
    feature_vectors = []

    threshold = 42

    # restore similar connected components
    region_groups = []

    # mark if one connected component is grouped
    is_grouped = []

    group_temp = []
    count = len(connected_components)

    # store weight value of each feature
    weights = [0.5, 0.5, 0.3, 1]

    # generate feature vectors
    for i in range(0, len(connected_components)):
        feature_vectors.append([area_list[i], perimeter_list[i], circularity1_list[i], circularity2_list[i]])
        is_grouped.append(False)

    # if count == 0, go out from the while loop
    while (count != 0):
        for i in range(0, len(feature_vectors) - 1):
            if is_grouped[i] == True:
                continue

            # if one connected component is grouped, the corresponding bool value in is_grouped is True
            is_grouped[i] = True
            count -= 1
            group_temp.append(i)

            for j in range(i + 1, len(feature_vectors)):
                if is_grouped[j] == False:
                    if Compute_Vector_Distance(feature_vectors[i], feature_vectors[j], weights) < threshold:
                        is_grouped[j] = True
                        count -= 1
                        group_temp.append(j)

            region_groups.append(group_temp.copy())
            group_temp.clear()

    for i in region_groups:
        print("Connected components with number", i, "are similar")



    ##### show rgb color image
    cv2.imshow('image_color', image_color)
    cv2.waitKey(0)
############ show processed images ############



