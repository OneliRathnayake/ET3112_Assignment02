import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# Load the cropped image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1(c).png", cv.IMREAD_GRAYSCALE)

# Display and save original image
plt.figure()
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.savefig("original_image.png", dpi=300)
plt.show()

# Apply Canny Edge Detector
edges = cv.Canny(img, 550, 690)

# Display and save edge image
plt.figure()
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.savefig("canny_edges.png", dpi=300)
plt.show()

# Extract edge coordinates
indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

# Scatter plot
plt.figure()
plt.scatter(x, y, s=1)
plt.title("Scatter Plot of Edge Points")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().invert_yaxis()

plt.savefig("scatter_plot.png", dpi=300)
plt.show()