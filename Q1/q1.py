import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 550, 690)

# MUST USE THIS PART
indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1)
plt.title("Scatter Plot of Edge Points")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().invert_yaxis()
plt.show()