###########################################################################################
#
#   This project provides some image process functions:
#       image enhancement functions:    Histogram_Stretching    Power_Law    Logarithm    Histogram_Equalization
#       image smooth functions:    Mean_Filter    Median_Filter    Gaussian_Filter
#       edge detector functions:    Canny    Marr_Hildreth    Sobel    Laplacian
#
#   This project uses Tkinter to build its GUI system
#       You can implement arbitrary functions in an arbitrary order
#
###########################################################################################
import cv2
import math
import time
import numpy as np
import tkinter as tk
import tkinter.filedialog
from tkinter import Canvas
from PIL import ImageTk, Image
import tkinter.messagebox as messagebox



################## region image process functions
################## region image extension functions
def Image_Extension(kernel_Size):
    global image_Process

    kernel_HalfSize = kernel_Size // 2

    height = image_Process.shape[0]
    width = image_Process.shape[1]

    image_EdgeExtension_Height = height + kernel_HalfSize * 2
    image_EdgeExtension_Width = width + kernel_HalfSize * 2

    image_EdgeExtension = np.zeros((image_EdgeExtension_Height, image_EdgeExtension_Width))

    # initialize values of extended image
    image_EdgeExtension[kernel_HalfSize: (image_EdgeExtension_Height - kernel_HalfSize), kernel_HalfSize: (image_EdgeExtension_Width - kernel_HalfSize)] = image_Process

    # handle extended column
    for i in range(kernel_HalfSize, (image_EdgeExtension_Width - kernel_HalfSize)):
        image_EdgeExtension[: kernel_HalfSize, i] = image_Process[0, (i - kernel_HalfSize)]
        image_EdgeExtension[-1 * (kernel_HalfSize):, i] = image_Process[(height - 1), (i - kernel_HalfSize)]

    # handle extended row
    for i in range(kernel_HalfSize, (image_EdgeExtension_Height - kernel_HalfSize)):
        image_EdgeExtension[i, : kernel_HalfSize] = image_Process[(i - kernel_HalfSize), 0]
        image_EdgeExtension[i, -1 * (kernel_HalfSize):] = image_Process[(i - kernel_HalfSize), (width - 1)]

    # handle extended diagonal
    image_EdgeExtension[: kernel_HalfSize, : kernel_HalfSize] = image_Process[0, 0]  # upper left
    image_EdgeExtension[: kernel_HalfSize, -1 * (kernel_HalfSize):] = image_Process[0, (width - 1)]  # upper right
    image_EdgeExtension[-1 * (kernel_HalfSize):, : kernel_HalfSize] = image_Process[(height - 1), 0]  # lower left
    image_EdgeExtension[-1 * (kernel_HalfSize):, -1 * (kernel_HalfSize):] = image_Process[(height - 1), (width - 1)]  # lower right

    return image_EdgeExtension
################## endregion


################## region image enhancement functions
def Histogram_Stretching(stretch_Min, stretch_Max):
    # use Histogram Stretching to map the old pixel value interval into [stretch_Min, stretch_Max]
    global image_Process

    grayMax = image_Process.max()
    grayMin = image_Process.min()

    coefficient = (stretch_Max - stretch_Min) / (grayMax - grayMin)

    # new_pixel_value = (stretch_Max - stretch_Min) / (grayMax - grayMin) * (old_pixel_value - grayMin) + stretch_Min
    image_Process = coefficient * (image_Process - grayMin) + stretch_Min

def Power_Law(coefficient, exponent):
    global image_Process

    # new_pixel_value = coefficient * (old_pixel_value / 255) ^ exponent
    image_Process = coefficient * (image_Process / 255) ** exponent

def Log(coefficient):
    global image_Process
    height, width = image_Process.shape

    # new_pixel_value = coefficient * log(old_pixel_value + 1)
    for row in range(0, height):
        for column in range(0, width):
            image_Process[row, column] = coefficient * math.log1p(image_Process[row, column])

def Histogram_Equalization(z_Max):
    global image_Process
    height, width = image_Process.shape

    # S is the total of pixels
    sum = height * width * 1.0
    sum_H = 0.0

    out = image_Process.copy()

    for i in range(1, 255):
        ind = np.where(image_Process == i)
        sum_H += len(image_Process[ind])
        z_prime = z_Max / sum * sum_H
        out[ind] = z_prime

    out = out.astype(np.uint8)
    image_Process = out
################## endregion


################## region image smooth functions
# use Mean Filter to smooth image
def Mean_Filter(kernel_Size):
    if kernel_Size % 2 == 0 or kernel_Size == 1:
        print('kernel size must be 3, 5, 7, 9....')
        return None

    global image_Process
    height, width = image_Process.shape

    # to calculate the convolution of boundary pixels, we have to extend the image
    image_Extension = Image_Extension(kernel_Size)
    kernel_Size_Square_Reciprocal  = 1 / (kernel_Size * kernel_Size)

    # calculate the convolution
    for row in range(0, height):
        for column in range(0, width):
            image_Process[row, column] = image_Extension[row: (row + kernel_Size), column: (column + kernel_Size)].sum() * kernel_Size_Square_Reciprocal

# use Median Filter to smooth image
def Median_Filter(kernel_Size):
    if kernel_Size % 2 == 0 or kernel_Size == 1:
        print('kernel size must be 3, 5, 7, 9....')
        return None

    global image_Process
    height, width = image_Process.shape

    # to calculate the convolution of boundary pixels, we have to extend the image
    image_Extension = Image_Extension(kernel_Size)
    kernel_Square = kernel_Size * kernel_Size
    kernel_Median = kernel_Square // 2

    for row in range(0, height):
        for column in range(0, width):
            line = np.sort(image_Extension[row: (row + kernel_Size), column: (column + kernel_Size)].flatten())
            image_Process[row, column] = line[kernel_Median]

# use Gaussian Filter to smooth image
def Gaussian_Filter(kernel_Size, sigma): # sigma is the standard deviation
    if kernel_Size % 2 == 0 or kernel_Size == 1:
        print('kernel size must be 3, 5, 7, 9....')
        return None

    global image_Process
    height, width = image_Process.shape
    image_Extension = Image_Extension(kernel_Size)

    ### region generate gaussian kernel
    kernel_HalfSize = kernel_Size // 2
    gaussian_kernel = np.ones((kernel_Size, kernel_Size))
    coefficient = 1 / (2 * (sigma ** 2))
    for row in range(0, kernel_Size):
        for column in range(0, kernel_Size):
            gaussian_kernel[row, column] = np.exp(-((column - kernel_HalfSize) ** 2 + (row - kernel_HalfSize) ** 2) * coefficient)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    ### endregion

    # use Gaussian Filter to calculate convolution of each pixel
    for row in range(0, height):
        for column in range(0, width):
            image_Process[row, column] = (image_Extension[row: (row + kernel_Size), column: (column + kernel_Size)] * gaussian_kernel).sum()
################## endregion

################## region edge detector functions
def Canny_Edge_Detector(threshold_Low, threshold_High):
    global image_Process
    height, width = image_Process.shape

    # calculate sobel filter, set the size of sobel filter 3
    sobel_Row = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_Column = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    image_Extension = Image_Extension(3)

    # store the data of image after getting handled with sobel filter
    image_Sobel = np.ones((height, width), dtype=np.uint8)

    # store the data of image after getting handled with Non-Maximum Suppression(NMS)
    image_NMS = np.zeros((height, width), dtype=np.uint8)

    # store the data of image after getting handled with Double threshold and Edge tracking by hysteresis
    image_Threshold = np.zeros((height, width), dtype=np.uint8)

    # store the direction of gradient of each pixel
    image_Theta = np.ones((height, width))

    pi_Div_2 = math.pi / 2  # 90 degree
    pi_Div_4 = math.pi / 4  # 45 degree

    ### region use sobel filter to calculate the convolution of each pixel
    for row in range(0, height):
        for column in range(0, width):
            gradient_Row = (image_Extension[row: (row + 3), column: (column + 3)] * sobel_Row).sum()
            gradient_Column = (image_Extension[row: (row + 3), column: (column + 3)] * sobel_Column).sum()

            image_Sobel[row, column] = math.sqrt(math.pow(gradient_Row, 2) + math.pow(gradient_Column, 2))

            if gradient_Column == 0 and gradient_Row == 0: # if there is no gradient, set image_Theta[row, column] = 777
                image_Theta[row, column] = 777
            elif gradient_Column == 0 and gradient_Row != 0: # if the degree is 90
                image_Theta[row, column] = pi_Div_2
            else:
                image_Theta[row, column] = math.atan(gradient_Row / gradient_Column)
    ### endregion

    ### region Non-Maximum Suppression(NMS)
    up = 0
    down = 0
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            weight = math.tan(image_Theta[row, column])

            # when gradient direction is between 90 and 45
            if image_Theta[row, column] >= pi_Div_4 and image_Theta[row, column] < pi_Div_2:
                weight = 1 / weight
                up = weight * image_Sobel[row - 1, column] + (1 - weight) * image_Sobel[row - 1, column + 1]
                down = weight * image_Sobel[row + 1, column] + (1 - weight) * image_Sobel[row + 1, column - 1]

            # when gradient direction is between 45 and 0
            elif image_Theta[row, column] >= 0 and image_Theta[row, column] < pi_Div_4:
                up = weight * image_Sobel[row, column + 1] + (1 - weight) * image_Sobel[row - 1, column + 1]
                down = weight * image_Sobel[row, column - 1] + (1 - weight) * image_Sobel[row + 1, column - 1]

            # when gradient direction is between 0 and -45
            elif image_Theta[row, column] < 0 and image_Theta[row, column] >= - pi_Div_4:
                weight = - weight
                up = weight * image_Sobel[row, column - 1] + (1 - weight) * image_Sobel[row - 1, column - 1]
                down = weight * image_Sobel[row, column + 1] + (1 - weight) * image_Sobel[row + 1, column + 1]

            # when gradient direction is between -45 and -90
            elif image_Theta[row, column] < - pi_Div_4 and image_Theta[row, column] > - pi_Div_2:
                weight = - 1 / weight
                up = weight * image_Sobel[row + 1, column] + (1 - weight) * image_Sobel[row + 1, column + 1]
                down = weight * image_Sobel[row - 1, column] + (1 - weight) * image_Sobel[row - 1, column - 1]

            # if the current pixel value is the maximum point
            if image_Sobel[row, column] >= up and image_Sobel[row, column] >= down:
                image_NMS[row, column] = 255
    ### endregion

    ### region Double threshold, we set low threshold and high threshold
    # supposed that all pixels of image_Threshold are weak edge pixels(set all pixel values to be 0 at beginning)
    for row in range(0, height):
        for column in range(0, width):
            if image_NMS[row, column] == 255:

                # if: current pixel value > high threshold, then: the pixel is a strong edge pixel
                if image_Sobel[row, column] > threshold_High:
                    image_Threshold[row, column] = 255

                # if: low threshold < current pixel value < high threshold, then: the pixel is a weak edge pixel
                if image_Sobel[row, column] > threshold_Low and image_Sobel[row, column] < threshold_High:
                    image_Threshold[row, column] = 100
    ### endregion

    ### region Edge tracking by hysteresis
    # use Depth First Search to find the weak edge pixels which have connection to the strong edge pixels
    stack_ = []     # store the pixels to be searched
    list_ = []      # store a list of pixels which have been searched in one deep search
    domain_8 = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            if image_Threshold[row, column] == 100:

                # if the weak pixel has benn searched, then mark the pixel by assign value of 101
                image_Threshold[row, column] = 101

                # if weak pixels in list_ have connection to a strong pixel, then connected_ = True
                connected_ = False

                stack_.append([row, column])
                list_.append([row, column])

                while (len(stack_) != 0):
                    # pos_center store the position of pixel currently under 8 domain search
                    pos_center = stack_.pop()

                    # 8 domain search to find the strong edge pixel
                    for pos in domain_8:
                        if image_Threshold[pos_center[0] + pos[0], pos_center[1] + pos[1]] == 255:
                            connected_ = True
                        if image_Threshold[pos_center[0] + pos[0], pos_center[1] + pos[1]] == 100:
                            image_Threshold[pos_center[0] + pos[0], pos_center[1] + pos[1]] = 101
                            stack_.append([pos_center[0] + pos[0], pos_center[1] + pos[1]])
                            list_.append([pos_center[0] + pos[0], pos_center[1] + pos[1]])

                # if weak pixels in list_ have no connection to a strong pixel, then assign corresponding pixel value to 0
                if connected_ == False:
                    for pos in list_:
                        image_Threshold[pos[0], pos[1]] = 0

                # if weak pixels in list_ have connection to a strong pixel, then assign corresponding pixel value to 0
                elif connected_ == True:
                    for pos in list_:
                        image_Threshold[pos[0], pos[1]] = 255

                stack_ = []
                list_ = []
    ### endregion

    image_Process = image_Threshold

def Marr_Hildreth_Edge_Detector(sigma):
    global image_Process
    height, width = image_Process.shape

    # determine the window size according to the Gaussian standard deviation of the input
    # 3 * Sigma accounted for 99.7%
    size = int(2 * (np.ceil(3 * sigma)) + 1)

    # Generate a grid of (-size / 2 + 1, size / 2)
    x, y = np.meshgrid(np.arange(- (size // 2), size // 2 + 1), np.arange(- (size // 2), size // 2 + 1))

    # calculate LoG kernel
    kernel_LoG = ((x ** 2 + y ** 2 - (2 * sigma ** 2)) / sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel_Size = kernel_LoG.shape[0]
    kernel_Half_Size = kernel_Size // 2

    image_Extension = Image_Extension(kernel_Size)

    # generate an all-zero matrix log of the same size as the input image
    image_LoG = np.zeros_like(image_Process, dtype=float)

    # apply the LoG core
    for row in range(0, height):
        for column in range(0, width):
            image_LoG[row, column] = (image_Extension[row:row+kernel_Size, column:column+kernel_Size]*kernel_LoG).sum()

    # convert type from float to int64
    image_LoG = image_LoG.astype(np.int64, copy=False)

    # image_ZC(Zero Crossing) stores zero crossing points
    image_ZC = np.zeros_like(image_Process)

    domain_8 = np.array([[-1, -1], [1, 0], [-1, 0], [1, 0], [-1, 1], [1, -1], [0, -1], [0, 1]])

    # find the zero crossing points
    for row in range(kernel_Half_Size, height - kernel_Half_Size):
        for column in range(kernel_Half_Size, width - kernel_Half_Size):

            # if current pixel valur = 0
            if image_LoG[row][column] == 0:
                for index in [0, 2, 4, 6]:
                    if (image_LoG[row + domain_8[index, 0]][column + domain_8[index, 0]]
                            * image_LoG[row + domain_8[index + 1, 0]][column + domain_8[index + 1, 0]] < 0):
                        image_ZC[row][column] = 255
                        break

            # if current pixel valur < 0
            elif image_LoG[row][column] < 0:
                for pos in domain_8:
                    if (image_LoG[row + pos[0]][column + pos[1]] > 0):
                        image_ZC[row][column] = 255
                        break

    image_Process = image_ZC

def Sobel_Edge_Detector(threshold_Low, threshold_High):
    global image_Process
    height, width = image_Process.shape

    # calculate sobel filter, set the size of sobel filter 3
    sobel_Row = np.array([[1, 3, 1], [0, 0, 0], [-1, -3, -1]]) # horrizontal edge
    sobel_Column = np.array([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]) # vertical edge

    image_Extension = Image_Extension(3)

    # store the data of image after getting handled with sobel filter
    image_Sobel = np.ones((height, width), dtype=np.uint8)

    # store the data of image after getting handled with Double threshold and Edge tracking by hysteresis
    image_Threshold = np.zeros((height, width), dtype=np.uint8)

    ### region use sobel filter to calculate the convolution of each pixel
    for row in range(0, height):
        for column in range(0, width):
            gradient_Row = (image_Extension[row: (row + 3), column: (column + 3)] * sobel_Row).sum()
            gradient_Column = (image_Extension[row: (row + 3), column: (column + 3)] * sobel_Column).sum()
            image_Sobel[row, column] = math.sqrt(math.pow(gradient_Row, 2) + math.pow(gradient_Column, 2))
    ### endregion

    ### region Double threshold, we set low threshold and high threshold
    # supposed that all pixels of image_Threshold are weak edge pixels(set all pixel values to be 0 at beginning)
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            # if: current pixel value > high threshold, then: the pixel is a strong edge pixel
            if image_Sobel[row, column] > threshold_High:
                image_Threshold[row, column] = 255

            # if: low threshold < current pixel value < high threshold, then: the pixel is a weak edge pixel
            elif image_Sobel[row, column] > threshold_Low and image_Sobel[row, column] < threshold_High:
                image_Threshold[row, column] = 100
    ### endregion

    ### region Edge tracking by hysteresis
    # use Depth First Search to find the weak edge pixels which have connection to the strong edge pixels
    stack_ = []     # store the pixels to be searched
    list_ = []      # store a list of pixels which have been searched in one deep search
    domain_8 = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            if image_Threshold[row, column] == 100:

                # if the weak pixel has benn searched, then mark the pixel by assign value of 101
                image_Threshold[row, column] = 101

                # if weak pixels in list_ have connection to a strong pixel, then connected_ = True
                connected_ = False

                stack_.append([row, column])
                list_.append([row, column])

                while (len(stack_) != 0):
                    # pos_center store the position of pixel currently under 8 domain search
                    pos_center = stack_.pop()
                    # 8 domain search to find the strong edge pixel
                    for pos in domain_8:
                        if image_Threshold[pos_center[0] + pos[0], pos_center[1] + pos[1]] == 255:
                            connected_ = True
                        elif image_Threshold[pos_center[0] + pos[0], pos_center[1] + pos[1]] == 100:
                            image_Threshold[pos_center[0] + pos[0], pos_center[1] + pos[1]] = 101
                            stack_.append([pos_center[0] + pos[0], pos_center[1] + pos[1]])
                            list_.append([pos_center[0] + pos[0], pos_center[1] + pos[1]])

                # if weak pixels in list_ have no connection to a strong pixel, then assign corresponding pixel value to 0
                if connected_ == False:
                    for pos in list_:
                        image_Threshold[pos[0], pos[1]] = 0

                # if weak pixels in list_ have connection to a strong pixel, then assign corresponding pixel value to 0
                elif connected_ == True:
                    for pos in list_:
                        image_Threshold[pos[0], pos[1]] = 255

                stack_ = []
                list_ = []
    ### endregion

    image_Process = image_Threshold

def Laplacian_Edge_Detector():
    global image_Process
    height, width = image_Process.shape

    # calculate laplace filter, set the size of sobel filter 3
    laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    image_Extension = Image_Extension(3)

    ### region use laplace filter to calculate the convolution of each pixel
    for row in range(0, height):
        for column in range(0, width):
            image_Process[row, column] = (image_Extension[row: (row + 3), column: (column + 3)] * laplacian).sum()
    ### endregion
################## endregion
################## endregion



################## region GUI functions
###### region other GUI functions
def Process_Record(function, time):
    global Process_Record_text
    Process_Record_text.insert('end', str(function))
    Process_Record_text.insert(tk.INSERT, '\n')
    Process_Record_text.insert('end', str(time))
    Process_Record_text.insert(tk.INSERT, '\n')
    Process_Record_text.insert(tk.INSERT, '\n')

def Choose_Image():
    global image_Origin
    global image_Process
    file_path = tk.filedialog.askopenfilename(title='Select an image.', filetypes=[('PGM', '*.pgm')])
    image_Process = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
    image_Process = cv2.cvtColor(image_Process, cv2.COLOR_RGB2GRAY)
    image_Origin = image_Process.copy()
    Show_Image()

def Reset_Image():
    global image_Origin
    global image_Process
    global Process_Record_text
    image_Process = image_Origin.copy()
    Process_Record_text.delete(0.0, tkinter.END)
    Show_Image()

def Show_Image():
    global image_Process
    global Image_Window
    global Image_Toshow
    global Image_Canvas

    Image_Window.geometry(str(image_Process.shape[1]) + 'x' + str(image_Process.shape[0]))
    image_Process = np.uint8(image_Process)
    Image_Toshow = ImageTk.PhotoImage(image=Image.fromarray(image_Process))
    Image_Canvas.create_image(0, 0, anchor='nw', image=Image_Toshow)

def Close_Image():
    if messagebox.showwarning("Quit", "Do you really wish to quit?"):
        exit(0)
###### endregion

###### region GUI image enhancement functions
def IEF_Histogram_Stretching(stretch_Min, stretch_Max):
    start = time.process_time()
    Histogram_Stretching(int(stretch_Min), int(stretch_Max))  # implement Histogram_Stretching() functions
    end = time.process_time()
    text1 = "Histogram_Stretching" + "(stretch_Min : " + stretch_Min + ", stretch_Max : " + stretch_Max + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def IEF_Power_Law(coefficient, exponent):
    start = time.process_time()
    Power_Law(float(coefficient), float(exponent))  # implement Power_Law() functions
    end = time.process_time()
    text1 = "Power_Law" + "(coefficient : " + coefficient + ", exponent : " + str(exponent) + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def IEF_Log(coefficient):
    start = time.process_time()
    Log(float(coefficient))  # implement Log() functions
    end = time.process_time()
    text1 = "Log" + "(coefficient : " + coefficient + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def IEF_Histogram_Equalization(z_Max):
    start = time.process_time()
    Histogram_Equalization(float(z_Max))  # implement Histogram_Equalization() functions
    end = time.process_time()
    text1 = "Histogram_Equalization" + "(param1 : " + z_Max + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()
###### endregion

###### region GUI image smooth functions
def ISF_Mean_Filter(kernel_Size):
    start = time.process_time()
    Mean_Filter(int(kernel_Size))  # implement Mean_Filter() functions
    end = time.process_time()
    text1 = "Mean_Filter" + "(kernel_size : " + kernel_Size + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def ISF_Median_Filter(kernel_Size):
    start = time.process_time()
    Median_Filter(int(kernel_Size))  # implement Median_Filter() functions
    end = time.process_time()
    text1 = "Median_Filter" + "(kernel_size : " + kernel_Size + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def ISF_Gaussian_Filter(kernel_Size, sigma):
    start = time.process_time()
    Gaussian_Filter(int(kernel_Size), float(sigma))  # implement Gaussian_Filter() functions
    end = time.process_time()
    text1 = "Gaussian_Filter" + "(kernel_size : " + kernel_Size + ", sigma : " + sigma + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()
###### endregion

###### region GUI edge detector functions
def EDF_Canny(threshold_Low, threshold_High):
    start = time.process_time()
    Canny_Edge_Detector(int(threshold_Low), int(threshold_High))  # implement Canny_Edge_Detector() functions
    end = time.process_time()
    text1 = "Canny" + "(threshold_low : " + threshold_Low + ", threshold_high : " + threshold_High + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def EDF_Marr_Hildreth(sigma):
    start = time.process_time()
    Marr_Hildreth_Edge_Detector(float(sigma))  # implement Marr_Hildreth_Edge_Detector() functions
    end = time.process_time()
    text1 = "Marr_Hildreth" + "(sigma : " + sigma + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def EDF_Sobel(threshold_Low, threshold_High):
    start = time.process_time()
    Sobel_Edge_Detector(int(threshold_Low), int(threshold_High))  # implement Sobel_Edge_Detector() functions
    end = time.process_time()
    text1 = "Sobel" + "(threshold_low : " + threshold_Low + ", threshold_high : " + threshold_High + ")"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()

def EDF_Laplacian():
    start = time.process_time()
    Laplacian_Edge_Detector()  # implement Laplacian_Edge_Detector() functions
    end = time.process_time()
    text1 = "Laplacian" + "()"
    text2 = "Time Consuming: " + str(end - start) + "s"  # compute time consuming about image process functions
    Process_Record(text1, text2)
    Show_Image()
###### endregion
################## endregion



if __name__ == "__main__":
    image_Process = None
    image_Origin = None

    ###### region GUI
    window = tk.Tk()
    window.title('Assignment 2')
    window.geometry('700x950')  # Set the size of the window (width x height)

    ###### region create five principal UI frames and corresponding subframes
    IEF_frame = tk.Frame(borderwidth=5, relief="sunken")  # image enhancement functions frame
    ISF_frame = tk.Frame(borderwidth=5, relief="sunken")  # image smooth functions frame
    EDF_frame = tk.Frame(borderwidth=5, relief="sunken")  # edge detector functions frame
    PR_frame = tk.Frame(borderwidth=5, relief="sunken")  # process record frame
    CRI_frame = tk.Frame(borderwidth=5, relief="sunken")  # choose and reset image frame

    IEF_frame.place(x=0, y=0, width=350, height=500)
    ISF_frame.place(x=0, y=500, width=350, height=400)
    EDF_frame.place(x=350, y=0, width=350, height=500)
    PR_frame.place(x=350, y=500, width=350, height=400)
    CRI_frame.place(x=0, y=900, width=700, height=50)

    IEF_subframe = []
    for i in range(0, 9):
        IEF_subframe.append(tk.Frame(IEF_frame, borderwidth=1, bg="#d8dde2"))
        IEF_subframe[i].pack(fill='x')
    (tk.Label(IEF_subframe[0], text="Image Enhancement Functions", font=("Calibri", 20), bg="#b3bcc6")).pack(fill="x")

    ISF_subframe = []
    for i in range(0, 7):
        ISF_subframe.append(tk.Frame(ISF_frame, borderwidth=1, bg="#d8dde2"))
        ISF_subframe[i].pack(fill='x')
    (tk.Label(ISF_subframe[0], text="Image Smooth Functions", font=("Calibri", 20), bg="#b3bcc6")).pack(fill="x")

    EDF_subframe = []
    for i in range(0, 9):
        EDF_subframe.append(tk.Frame(EDF_frame, borderwidth=1, bg="#d8dde2"))
        EDF_subframe[i].pack(fill='x')
    (tk.Label(EDF_subframe[0], text="Edge Detector Functions", font=("Calibri", 20), bg="#b3bcc6")).pack(fill="x")
    ###### endregion

    ################## region image enhancement functions frame
    ###### region set GUI about Histogram Stretching
    (tk.Label(IEF_subframe[1], text="Histogram Stretching:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(IEF_subframe[2], text="stretch min", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(IEF_subframe[2], text="stretch max", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=1)
    stretch_Min_HS = tk.Entry(IEF_subframe[2], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="0"))
    stretch_Min_HS.grid(row=1, column=0)
    stretch_Max_HS = tk.Entry(IEF_subframe[2], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="255"))
    stretch_Max_HS.grid(row=1, column=1)
    Histogram_Stretching_button = (tk.Button(IEF_subframe[2], text="Process", font=("Calibri", 15),
        command=lambda: IEF_Histogram_Stretching(stretch_Min_HS.get(), stretch_Max_HS.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Power Law
    (tk.Label(IEF_subframe[3], text="Power Law:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(IEF_subframe[4], text="coefficient", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(IEF_subframe[4], text="exponent", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=1)
    coefficient_PL = tk.Entry(IEF_subframe[4], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="255"))
    coefficient_PL.grid(row=1, column=0)
    exponent_PL = tk.Entry(IEF_subframe[4], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="0.5"))
    exponent_PL.grid(row=1, column=1)
    Power_Law_button = (tk.Button(IEF_subframe[4], text="Process", font=("Calibri", 15),
        command=lambda: IEF_Power_Law(coefficient_PL.get(), exponent_PL.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Log
    (tk.Label(IEF_subframe[5], text="Log:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(IEF_subframe[6], text="coefficient", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(IEF_subframe[6], text="                       ", font=("Calibri", 15), bg="#d8dde2")).grid(row=1, column=1)
    coefficient_Log = tk.Entry(IEF_subframe[6], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="100"))
    coefficient_Log.grid(row=1, column=0)
    Log_button = (tk.Button(IEF_subframe[6], text="Process", font=("Calibri", 15),
        command=lambda: IEF_Log(coefficient_Log.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Histogram Equalization
    (tk.Label(IEF_subframe[7], text="Histogram Equalization:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(IEF_subframe[8], text="z_Max", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(IEF_subframe[8], text="                       ", font=("Calibri", 15), bg="#d8dde2")).grid(row=1, column=1)
    z_Max = tk.Entry(IEF_subframe[8], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="255"))
    z_Max.grid(row=1, column=0)
    Histogram_Equalization_button = (tk.Button(IEF_subframe[8], text="Process", font=("Calibri", 15),
        command=lambda: IEF_Histogram_Equalization(z_Max.get()))).grid(row=1, column=2)
    ###### endregion
    ################## endregion

    ################## region image smooth functions frame
    ###### region set GUI about Mean Filter
    (tk.Label(ISF_subframe[1], text="Mean Filter:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(ISF_subframe[2], text="kernel size", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(ISF_subframe[2], text="                       ", font=("Calibri", 15), bg="#d8dde2")).grid(row=1, column=1)
    kernel_Size_Mean = tk.Entry(ISF_subframe[2], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="3"))
    kernel_Size_Mean.grid(row=1, column=0)
    Mean_Filter_button = (tk.Button(ISF_subframe[2], text="Process", font=("Calibri", 15),
        command=lambda: ISF_Mean_Filter(kernel_Size_Mean.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Median Filter
    (tk.Label(ISF_subframe[3], text="Median Filter:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(ISF_subframe[4], text="kernel size", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(ISF_subframe[4], text="                       ", font=("Calibri", 15), bg="#d8dde2")).grid(row=1, column=1)
    kernel_Size_Median = tk.Entry(ISF_subframe[4], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="3"))
    kernel_Size_Median.grid(row=1, column=0)
    Median_Filter_button = (tk.Button(ISF_subframe[4], text="Process", font=("Calibri", 15),
        command=lambda: ISF_Median_Filter(kernel_Size_Median.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Gaussian Filter
    (tk.Label(ISF_subframe[5], text="Gaussian Filter:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(ISF_subframe[6], text="kernel Size", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(ISF_subframe[6], text="sigma", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=1)
    kernel_Size_GF = tk.Entry(ISF_subframe[6], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="3"))
    kernel_Size_GF.grid(row=1, column=0)
    sigma_GF = tk.Entry(ISF_subframe[6], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="0.8"))
    sigma_GF.grid(row=1, column=1)
    Gaussian_Filter_button = (tk.Button(ISF_subframe[6], text="Process", font=("Calibri", 15),
        command=lambda: ISF_Gaussian_Filter(kernel_Size_GF.get(), sigma_GF.get()))).grid(row=1, column=2)
    ###### endregion
    ################## endregion

    ################## region edge detector functions frame
    ###### region set GUI about Canny edge detector
    (tk.Label(EDF_subframe[1], text="Canny:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(EDF_subframe[2], text="threshold low", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(EDF_subframe[2], text="threshold high", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=1)
    threshold_Low_Canny = tk.Entry(EDF_subframe[2], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="50"))
    threshold_Low_Canny.grid(row=1, column=0)
    threshold_High_Canny = tk.Entry(EDF_subframe[2], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="100"))
    threshold_High_Canny.grid(row=1, column=1)
    Canny_button = (tk.Button(EDF_subframe[2], text="Process", font=("Calibri", 15),
        command=lambda: EDF_Canny(threshold_Low_Canny.get(), threshold_High_Canny.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Marr Hildreth
    (tk.Label(EDF_subframe[3], text="Marr Hildreth:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(EDF_subframe[4], text="sigma", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(EDF_subframe[4], text="                        ", font=("Calibri", 15), bg="#d8dde2")).grid(row=1, column=1)
    sigma_MH = tk.Entry(EDF_subframe[4], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="0.8"))
    sigma_MH.grid(row=1, column=0)
    Marr_Hildreth_button = (tk.Button(EDF_subframe[4], text="Process", font=("Calibri", 15),
        command=lambda: EDF_Marr_Hildreth(sigma_MH.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Sobel
    (tk.Label(EDF_subframe[5], text="Sobel:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(EDF_subframe[6], text="threshold low", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(EDF_subframe[6], text="threshold high", font=("Calibri", 15), bg="#d8dde2")).grid(row=0, column=1)
    threshold_Low_Sobel = tk.Entry(EDF_subframe[6], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="80"))
    threshold_Low_Sobel.grid(row=1, column=0)
    threshold_High_Sobel = tk.Entry(EDF_subframe[6], width=12, font=("Calibri", 15), textvariable=tk.StringVar(value="200"))
    threshold_High_Sobel.grid(row=1, column=1)
    Sobel_button = (tk.Button(EDF_subframe[6], text="Process", font=("Calibri", 15),
        command=lambda: EDF_Sobel(threshold_Low_Sobel.get(), threshold_High_Sobel.get()))).grid(row=1, column=2)
    ###### endregion

    ###### region set GUI about Laplacian
    (tk.Label(EDF_subframe[7], text="Laplacian:", font=("Calibri", 17))).pack(fill="x")
    (tk.Label(EDF_subframe[8], text="threshold low", font=("Calibri", 15), fg="#d8dde2", bg="#d8dde2")).grid(row=0, column=0)
    (tk.Label(EDF_subframe[8], text="threshold high", font=("Calibri", 15), fg="#d8dde2", bg="#d8dde2")).grid(row=0, column=1)
    Laplacia_button = (tk.Button(EDF_subframe[8], text="Process", font=("Calibri", 15),
        command=lambda: EDF_Laplacian())).grid(row=1, column=2)
    ###### endregion
    ################## endregion

    ################## region process record frame
    (tk.Label(PR_frame, text="Process Record", font=("Calibri", 20), bg="#b3bcc6")).pack(fill='x')
    # record the process of image processing
    Process_Record_text = tk.Text(PR_frame, width=30, height=60, font=("Calibri", 16))
    Process_Record_text.pack(fill='x')
    ################## endregion

    ################## region sub window to show and reset image
    Image_Window = tk.Toplevel()
    Image_Window.protocol("WM_DELETE_WINDOW", Close_Image)
    Image_Window.geometry('500x500+300+200')
    Image_Window.title("Image_Window")
    Image_Window.configure(background="white")

    Image_Toshow = None
    Image_Canvas = Canvas(Image_Window, bg='pink')
    Image_Canvas.place(x=0, y=0, width=800, height=600)

    # 'Choose Image' button
    Choose_Image_button = tk.Button(CRI_frame,
                                    text="Choose Image",
                                    font=("Calibri", 15),
                                    command=lambda: Choose_Image())
    Choose_Image_button.place(x = 200, y = 0)

    # 'Reset Image' button, reset current image
    Reset_Image_button = tk.Button(CRI_frame,
                                   text="Reset Image",
                                   font=("Calibri", 15),
                                   command=lambda: Reset_Image())
    Reset_Image_button.place(x = 360, y = 0)
    ################## endregion

    window.mainloop()
    ###### endregion




