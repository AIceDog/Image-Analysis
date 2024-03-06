# Mean Shift Segmentation

## 1. Brief Problem Definition
In this project, the Mean Shift technique with image segmentation is implemented and the experiment is conducted to verify the effectiveness of this algorithm. Mean shift is a non-parametric feature-space analysis technique that assigns the data points to the clusters iteratively by shifting points toward the mode. For the image segmentation problem, the pixel can be mapped to the color feature space. After that, the algorithm iteratively assigns each pixel to the closest cluster centroid according to the surrounding pixels. 

## 2. Summary of Choices Made for the Solution
In this project, the algorithm consists of 3 steps for each iteration which are mean shift filtering, merging similar clusters, and eliminating Small Regions.

### 2.1 Mean Shift Filtering
Mean shift filtering is used to find the closet cluster centroid for each pixel and replace the color of the pixel with the color of the cluster center to smooth the image.
In this experiment, the Gaussian kernel function to implement mean shift filtering is used for mean shift filtering. 
The multivariate kernel can be defined as the product of two radially symmetric kernels with two bandwidth parameters for each domain. Here is the form of the Gaussian kernel function:
<img src="imgs/3.1.jpg" height="230px">

### 2.2 Merge Similar Clusters
### 2.3 Eliminate Small Regions

## 3. Segmentation Results Of 2 Color And 2 Gray Scale Images

## 4. Brief Discussion Of Results

## 5. Other Examples

