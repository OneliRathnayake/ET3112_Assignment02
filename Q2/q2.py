import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the cropped image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detector
edges = cv.Canny(img, 550, 690)

# Extract edge coordinates (given in assignment)
indices = np.where(edges != [0])

x = indices[1]
y = indices[0]

# Scatter plot of edge points
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1)

plt.title("Scatter Plot of Extracted Edge Points")
plt.xlabel("x")
plt.ylabel("y")

# Invert y-axis to match image coordinate system
plt.gca().invert_yaxis()

plt.show()